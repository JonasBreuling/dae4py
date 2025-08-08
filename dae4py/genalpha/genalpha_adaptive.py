import numpy as np
from tqdm import tqdm
from scipy._lib._util import _RichResult
from scipy.integrate._ivp.common import EPS
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import splu


# TODO: Implement dense output using cubic hermite polynomial and
#       add extrapolation for better initial guess.
def solve_dae_genalpha_adaptive(
    F,
    y0,
    yp0,
    t_span,
    h0=1e-3,
    rho_inf=0.5,
    t_eval=None,
    atol=1e-6,
    rtol=1e-3,
    kappa=1.0,
    extrapolate_dense_output=False,
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
    rho_inf: float, default: 0.5
        Dampiing at infinity, 0 <= rho_inf <= 1.
    atol: float, defaul: 1e-6
        Absolute tolerance for the step-size controller.
    rtol: float, default: 1e-6
        Relative tolerance for the step-size controller.
    kappa: float, default: 1.0
        Scalng factor of the smooth limiter.
    extrapolate_dense_output: boolean, defaul: False
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
        - x (array-like): Auxiliary derivative values at the time grid.
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

    # dense output is not implemented
    assert t_eval is None
    assert not extrapolate_dense_output

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
    newton_max_iter = 7

    # parameters
    assert 0 <= rho_inf <= 1
    alpha_f = rho_inf / (rho_inf + 1)
    alpha_m = (3 * rho_inf - 1) / (2 * (rho_inf + 1))
    gamma = 0.5 + alpha_f - alpha_m
    mu = gamma * (1 - alpha_f) / (1 - alpha_m)

    # prepare initial values
    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)

    # initial guess for auxiliary derivative
    # TODO: Find second-order guess
    x0 = yp0.copy()

    # # initial guess for stage derivatives
    # Yp = np.tile(yp0, s).reshape(s, -1)
    # Y = y0 + h0 * A.dot(Yp)

    # initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]
    xs = [x0]

    hn = h0
    tn = t0
    yn = y0
    ypn = yp0
    xn = x0
    hn_old = None
    LU = None
    error_norm_old = None
    current_jac = True
    with tqdm(total=100, desc="Generalized-alpha") as pbar:
        while tn < t1:
            # ensure that last step exactly hits t1
            if (tn + hn - t1) > 0:
                hn = t1 - tn

            if hn > max_step:
                hn = max_step

            step_accepted = False
            while not step_accepted:
                if not extrapolate_dense_output:
                    ypn1 = ypn

                # simplified Newton iterations
                newton_scale = atol + np.abs(yn) * rtol
                converged = False
                while not converged:
                    # estimate Jacobians and compute factorizations
                    if LU is None:
                        LU = factor_lu(hn * mu * J + M)

                    # future time
                    tn1 = tn + hn

                    dy_norm_old = None
                    rate = None
                    for k in range(newton_max_iter):
                        # auxiliary derivative and state value
                        xn1 = (alpha_f * ypn + (1 - alpha_f) * ypn1 - alpha_m * xn) / (
                            1 - alpha_m
                        )
                        yn1 = yn + hn * ((1 - gamma) * xn + gamma * xn1)

                        # residual
                        FF = fun(tn1, yn1, ypn1)

                        dy_dot = solve_lu(LU, -FF)
                        dy = hn * mu * dy_dot

                        ypn1 += dy_dot

                        dy_norm = (
                            np.linalg.norm(dy / newton_scale) / newton_scale.size**0.5
                        )
                        if dy_norm_old is not None:
                            rate = dy_norm / dy_norm_old

                        if rate is not None and rate >= 1:
                            break

                        if (
                            rate is not None
                            and rate / (1 - rate) * dy_norm < newton_tol
                        ):
                            converged = True
                            break

                        dy_norm_old = dy_norm

                    if not converged:
                        if current_jac:
                            break

                        M, J = jac(tn, yn, ypn)
                        current_jac = True
                        LU = None

                if not converged:
                    hn *= 0.5
                    LU = None
                    continue

                # error estimate (w.r.t euler step)
                error = yn1 - (yn + hn * ypn1)
                scale = atol + np.maximum(np.abs(yn), np.abs(yn1)) * rtol
                error_norm = np.linalg.norm(error / scale) / scale.size**0.5

                # step-size control
                # p = 2, p_hat = 1 => p_hat + 1 = 2
                if error_norm_old is None or hn_old is None or error_norm == 0:
                    multiplier = 1
                else:
                    multiplier = hn / hn_old * (error_norm_old / error_norm) ** (1 / 2)

                with np.errstate(divide="ignore"):
                    factor = min(1, multiplier) * error_norm ** (-1 / 2)

                # smooth limiter
                factor = 1 + kappa * np.arctan((factor - 1) / kappa)

                # add safety factor
                safety = 0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + k + 1)
                factor *= safety

                # can the step be accepted
                if error_norm > 1:
                    hn *= factor
                    LU = None
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
                LU = None

            # append to solution arrays
            nsteps += 1
            t.append(tn1)
            y.append(yn1.copy())
            yp.append(ypn1.copy())
            xs.append(xn1.copy())

            # # initial guess for next iteration by extrapolating
            # # collocation polynomial
            # Z = Y - yn
            # ZTQT = Z.T @ Q.T
            # if extrapolate_dense_output:
            #     theta = 1 + c * factor
            #     exponent = np.arange(1, s + 1)[:, None]
            #     theta_hat_vec = exponent * theta ** (exponent - 1)
            #     Yp = (ZTQT @ (theta_hat_vec / hn)).T

            # # dense output
            # if t_eval is not None:
            #     t_eval_i1 = np.searchsorted(t_eval, tn + hn, side="right")
            #     t_eval_step = t_eval[t_eval_i:t_eval_i1]

            #     if t_eval_step.size > 0:
            #         t_eval_i = t_eval_i1

            #         theta = (t_eval_step - tn) / hn
            #         exponent = np.arange(1, s + 1)[:, None]
            #         theta_vec = theta**exponent
            #         theta_hat_vec = exponent * theta ** (exponent - 1)

            #         y_eval_step = yn[:, None] + ZTQT @ theta_vec
            #         yp_eval_step = ZTQT @ (theta_hat_vec / hn)

            #         y_eval.append(y_eval_step.T)
            #         yp_eval.append(yp_eval_step.T)

            # update old values
            tn = tn1
            yn = yn1.copy()
            ypn = ypn1.copy()
            xn = xn1.copy()
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
        x=np.array(xs),
        t_eval=t_eval,
        y_eval=np.concatenate(y_eval) if y_eval is not None else y_eval,
        yp_eval=np.concatenate(yp_eval) if y_eval is not None else yp_eval,
        nsteps=nsteps,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
        nlgs=nlgs,
    )
