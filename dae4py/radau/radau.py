import numpy as np
from tqdm import tqdm
from scipy._lib._util import _RichResult
from scipy.integrate._ivp.common import EPS
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import eig, cdf2rdf
from scipy.sparse.linalg import splu
from dae4py.butcher_tableau import radau_tableau


def solve_dae_radau(
    F,
    y0,
    yp0,
    t_span,
    h0=1e-3,
    s=3,
    t_eval=None,
    atol=1e-6,
    rtol=1e-3,
    kappa=1.0,
    eta=0.05,
    newton_iter_embedded=1,
    extrapolate_dense_output=True,
    jac=None,
    controller_deadband=(1.0, 1.2),
    jac_recompute_rate=1e-3,
    jac_recompute_newton_iter=2,
    max_step=np.infty,
):
    """
    Solves a system of DAEs using implicit Runge-Kutta methods with variable step-sizes.

    Parameters
    ----------
    F: callable
        Function defining the DAE system, F(t, y, yp) = 0.
    y0: array-like
        Initial condition for y.
    yp0: array-like
        Initial condition for y'.
    t_span: Tuple
        (t0, t1) defining the time span.
    h0: float, default: 1e-3
        Initial step-size.
    t_eval, array-like, optional
        Array of evaluation points for dense output.
    s: int, default: 3
        Number of stages (has to be odd).
    atol: float, defaul: 1e-6
        Absolute tolerance for the step-size controller.
    rtol: float, default: 1e-6
        Relative tolerance for the step-size controller.
    kappa: float, default: 1.0
        Scalng factor of the smooth limiter.
    eta: float, default: 0.05
        Absolute value of the stability function of the embedded method at
        infinity. Is only used when newton_iter_embedded >= 1, see below.
    newton_iter_embedded: int, default: 1
        Number of Newton iterations for embedded method:
            - 0: Explicit embedded method.
            - 1: Implicit embedded method with a single Newton iteration.
            - otherwise: Implicit embedded method with arbitrary number of
              Newton iterations.
    extrapolate_dense_output: boolean, defaul: True
        Use dense output function to extrapolate a new initial guess for the
        next time step.
    jac: {callable, None}, default: None
        Function that computes the Jacobian matrices M = dF/dy' and J = dF/dy.
        There are two different possibilities:
            * If callable, the Jacobians are assumed to depend on both
              t, y and y'; it will be called as ``M, J = jac(t, y, yp)``.
            * If None (default), the Jacobians will be approximated by
              finite differences using scipy's ``approx_derivative`` function.
    controller_deadband: tuple, defaul: [1.0, 1.2]
        Range of the step-size scaling factor for which we supress step-size
        changes in order to increase performance by not recomputing the LU
        decompositions.
    jac_recompute_rate: float, defaul: 1e-3
        Worst case rate of convergence that allows to reuse the Jacobian
        in the next step. This and the condition below have to be met.
    jac_recompute_newton_iter: int, defaul: 2
        Worst case number of newton iterations that allows to reuse the
        Jacobian in the next step. This and the condition above have to be met.

    Returns
    -------
    solution: _RichResult
        Container that stores
        - t (array-like): Time grid.
        - y (array-like): State at the time grid.
        - yp (array-like): Derivative at the time grid.
        - Y (array-like): Stage values at the time grid.
        - Yp (array-like): Stage derivative at the time grid.
        - t_eval (array-like): Time grid (dense output).
        - y_eval (array-like): State (dense output).
        - yp_eval (array-like): Derivative (dense output).
        - nsteps (int): Number of steps.
        - nfev (int): Number of function evaluations.
        - njev (int): Number of Jacobian evaluations.
        - nlu (int): Number of LU decompositions.
        - nlgs (int): Number of forward + backward substitutions to solve a
          linear system of equations with given LU-decompositions.
    """
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    if t_eval is not None:
        t_eval_i = 0
        t_eval = np.asarray(t_eval)
        y_eval = []
        yp_eval = []
    else:
        y_eval = None
        yp_eval = None

    # wrap function calls
    nsteps = 0
    nfev = 0
    njev = 0
    nlu = 0
    nlgs = 0

    def fun(t, y, yp):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(F(t, y, yp))

    if jac is None:

        def jac(t, y, yp):
            nonlocal njev
            njev += 1
            J = approx_derivative(lambda _y: F(t, _y, yp), y)
            M = approx_derivative(lambda _yp: F(t, y, _yp), yp)
            return M, J

    else:
        jac_ = jac

        def jac(t, y, yp):
            nonlocal njev
            njev += 1
            return jac_(t, y, yp)

    def factor_lu(A):
        nonlocal nlu
        nlu += 1
        return splu(A)

    def solve_lu(LU, rhs):
        nonlocal nlgs
        nlgs += 1
        return LU.solve(rhs)

    # initial Jacobians
    M, J = jac(t0, y0, yp0)

    # newton tolerance as in radau.f line 1008ff
    # newton_tol = max(10 * EPS / rtol, min(rtol**0.5, 0.03))
    # newton tolerance as in pside.f
    newton_tol = 0.01

    # maximum number of newton iterations:
    # - radau.f line 446 initially choses NIT=7 and subsequently updates the
    #   value using the formula below
    # - radaup.f line 416 choses NIT=7+(NS-3)*2
    # - pside.f line 1887 choses KMAX = 15
    newton_max_iter = 7 + int((s - 3) * 2.5)

    # Butcher tableau
    assert s % 2 == 1
    tableau = radau_tableau(s)
    A, b, c, s = tableau.A, tableau.b, tableau.c, tableau.s

    # eigenvalues and corresponding eigenvectors of coefficient matrix
    lambdas, U = eig(A)

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    U = U[:, idx]

    # convert complex eigenvalues and eigenvectors to real eigenvalues
    # in a block diagonal form and the associated real eigenvectors
    Gammas, T = cdf2rdf(lambdas, U)
    TI = np.linalg.inv(T)

    # sanity checks
    assert np.allclose(U @ np.diag(lambdas) @ np.linalg.inv(U), A)
    assert np.allclose(np.linalg.inv(U) @ A @ U, np.diag(lambdas))
    assert np.allclose(T @ Gammas @ TI, A)
    assert np.allclose(TI @ A @ T, Gammas)

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    gamma = gammas[0]
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag

    # extended quadrature nodes and Vandermonde matrix
    c_hat = np.array([0, *c])
    Vc_hat = np.vander(c_hat, increasing=True)

    # quadrature weights for explicit embedded method
    rhs = 1 / np.arange(1, s + 1)
    b_hat_1 = gamma
    rhs[0] -= b_hat_1
    b_hat = np.linalg.solve(Vc_hat.T[:-1, 1:], rhs)

    # quadrature weights for implicit embedded method
    rhs = 1 / np.arange(1, s + 1)
    b_tilde_s2 = gamma
    b_tilde_1 = eta * b_tilde_s2
    rhs[0] -= b_tilde_1
    rhs -= b_tilde_s2
    b_tilde = np.linalg.solve(Vc_hat.T[:-1, 1:], rhs)

    # compute the inverse Vandermonde matrix to get the inverse
    # interpolation matrix
    Q = np.linalg.inv(Vc_hat[1:, 1:])

    # prepare initial values
    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
    m = len(y0)

    # initial guess for stage derivatives
    Yp = np.tile(yp0, s).reshape(s, -1)
    Y = y0 + h0 * A.dot(Yp)

    # initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]
    Ys = [Y]
    Yps = [Yp]

    hn = h0
    tn = t0
    yn = y0
    ypn = yp0
    hn_old = None
    error_norm_old = None
    current_jac = True
    LU_real = None
    LU_complex = None
    with tqdm(total=100, desc="Radau IIA") as pbar:
        while tn < t1:
            # ensure that last step exactly hits t1
            if (tn + hn - t1) > 0:
                hn = t1 - tn

            if hn > max_step:
                hn = max_step

            step_accepted = False
            while not step_accepted:
                if not extrapolate_dense_output:
                    Yp = np.tile(ypn, s).reshape(s, -1)

                # simplified Newton iterations
                newton_scale = atol + np.abs(yn) * rtol
                converged = False
                while not converged:
                    # estimate Jacobians and compute factorizations
                    if LU_real is None or LU_complex is None:
                        LU_real = factor_lu(M + hn * gamma * J)
                        LU_complex = [
                            factor_lu(M + hn * (alpha - 1j * beta) * J)
                            for (alpha, beta) in zip(alphas, betas)
                        ]

                    # quadrature
                    tau = tn + c * hn
                    Y = yn + hn * A.dot(Yp)

                    Fs = np.empty((s, m))
                    dY_norm_old = None
                    dW_dot = np.empty_like(Fs)
                    rate = None
                    s_complex = len(LU_complex)
                    for k in range(newton_max_iter):
                        for i in range(s):
                            Fs[i] = fun(tau[i], Y[i], Yp[i])

                        G = TI @ Fs
                        G_real = -G[0]
                        G_complex = np.empty((s_complex, m), dtype=complex)
                        for i in range(s_complex):
                            G_complex[i] = -(G[2 * i + 1] + 1j * G[2 * i + 2])

                        dW_dot_real = solve_lu(LU_real, G_real)
                        dW_dot_complex = np.empty_like(G_complex)
                        for i in range(s_complex):
                            dW_dot_complex[i] = solve_lu(LU_complex[i], G_complex[i])

                        dW_dot[0] = dW_dot_real
                        for i in range(s_complex):
                            dW_dot[2 * i + 1] = dW_dot_complex[i].real
                            dW_dot[2 * i + 2] = dW_dot_complex[i].imag

                        dYp = T.dot(dW_dot)
                        dY = hn * A.dot(dYp)

                        Yp += dYp
                        Y += dY

                        dY_norm = (
                            np.linalg.norm(dY / newton_scale) / newton_scale.size**0.5
                        )
                        if dY_norm_old is not None:
                            rate = dY_norm / dY_norm_old

                        if rate is not None and rate >= 1:
                            break

                        if (
                            rate is not None
                            and rate / (1 - rate) * dY_norm < newton_tol
                        ):
                            converged = True
                            break

                        dY_norm_old = dY_norm

                    if not converged:
                        if current_jac:
                            break

                        M, J = jac(tn, yn, ypn)
                        current_jac = True
                        LU_real = None
                        LU_complex = None

                if not converged:
                    hn *= 0.5
                    LU_real = None
                    LU_complex = None
                    continue

                # stiffly accurate method
                # tn1 = tau[-1] # this leads to roundoff errors
                tn1 = tn + hn
                yn1 = Y[-1]
                ypn1 = Yp[-1]

                # error estimate
                match newton_iter_embedded:
                    case 0:
                        error = hn * ((b - b_hat) @ Yp - b_hat_1 * ypn)
                    case 1:
                        yp_tilde = ((b - b_tilde) @ Yp - b_tilde_1 * ypn) / b_tilde_s2
                        F_tilde = fun(tn1, yn1, yp_tilde)
                        error = hn * b_tilde_s2 * solve_lu(LU_real, F_tilde)
                    case _:
                        yp_tilde0 = (
                            -(yn / hn + b_tilde_1 * ypn + b_tilde @ Yp) / b_tilde_s2
                        )
                        y_tilde = yn1.copy()  # initial guess
                        for _ in range(newton_iter_embedded):
                            yp_tilde = yp_tilde0 + y_tilde / (hn * b_tilde_s2)
                            F_tilde = fun(tn1, y_tilde, yp_tilde)
                            y_tilde -= hn * b_tilde_s2 * solve_lu(LU_real, F_tilde)

                        error = yn1 - y_tilde

                scale = atol + np.maximum(np.abs(yn), np.abs(yn1)) * rtol
                error_norm = np.linalg.norm(error / scale) / scale.size**0.5

                # step-size control
                if error_norm_old is None or hn_old is None or error_norm == 0:
                    multiplier = 1
                else:
                    multiplier = (
                        hn / hn_old * (error_norm_old / error_norm) ** (1 / (s + 1))
                    )

                with np.errstate(divide="ignore"):
                    factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

                # smooth limiter
                factor = 1 + kappa * np.arctan((factor - 1) / kappa)

                # add safety factor
                safety = 0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + k + 1)
                factor *= safety

                # can the step be accepted
                if error_norm > 1:
                    hn *= factor
                    LU_real = None
                    LU_complex = None
                else:
                    step_accepted = True

            # compute new Jacobian if convergence is too slow
            recompute_jac = (
                k + 1 > jac_recompute_newton_iter and rate > jac_recompute_rate
            )

            # possibly do not alter step-size
            if (
                not recompute_jac
                and controller_deadband[0] <= factor <= controller_deadband[1]
            ):
                factor = 1
                current_jac = False
            else:
                M, J = jac(tn1, yn1, ypn1)
                current_jac = True
                LU_real = None
                LU_complex = None

            # append to solution arrays
            nsteps += 1
            t.append(tn1)
            y.append(yn1.copy())
            yp.append(ypn1.copy())
            Ys.append(Y.copy())
            Yps.append(Yp.copy())

            # initial guess for next iteration by extrapolating
            # collocation polynomial
            Z = Y - yn
            ZTQT = Z.T @ Q.T
            if extrapolate_dense_output:
                theta = 1 + c * factor
                exponent = np.arange(1, s + 1)[:, None]
                theta_hat_vec = exponent * theta ** (exponent - 1)
                Yp = (ZTQT @ (theta_hat_vec / hn)).T

            # dense output
            if t_eval is not None:
                t_eval_i1 = np.searchsorted(t_eval, tn + hn, side="right")
                t_eval_step = t_eval[t_eval_i:t_eval_i1]

                if t_eval_step.size > 0:
                    t_eval_i = t_eval_i1

                    theta = (t_eval_step - tn) / hn
                    exponent = np.arange(1, s + 1)[:, None]
                    theta_vec = theta**exponent
                    theta_hat_vec = exponent * theta ** (exponent - 1)

                    y_eval_step = yn[:, None] + ZTQT @ theta_vec
                    yp_eval_step = ZTQT @ (theta_hat_vec / hn)

                    y_eval.append(y_eval_step.T)
                    yp_eval.append(yp_eval_step.T)

            # update old values
            tn = tn1
            yn = yn1
            ypn = ypn1
            hn_old = hn
            error_norm_old = error_norm

            # update progress bar
            progress = min(100, int(100 * (tn - t0) / (t1 - t0)))
            pbar.n = progress
            pbar.set_description(f"t: {tn:0.2e}s < {t1:0.2e}s; h: {hn:0.2e}")
            pbar.refresh()

            # fianlly update the step-size for the next step
            hn *= factor

    return _RichResult(
        t=np.array(t),
        y=np.array(y),
        yp=np.array(yp),
        Y=np.array(Ys),
        Yp=np.array(Yps),
        t_eval=t_eval,
        y_eval=np.concatenate(y_eval) if y_eval is not None else y_eval,
        yp_eval=np.concatenate(yp_eval) if y_eval is not None else yp_eval,
        nsteps=nsteps,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
        nlgs=nlgs,
    )
