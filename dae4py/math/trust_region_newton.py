import numpy as np
from scipy._lib._util import _RichResult
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import solve as linear_solve

EPS = np.finfo(float).eps


def _dogleg_step(p_newton, p_cauchy, radius):
    """Powell's dogleg path, truncated to the trust region of given radius."""
    norm_newton = np.linalg.norm(p_newton)
    if norm_newton <= radius:
        return p_newton, False

    norm_cauchy = np.linalg.norm(p_cauchy)
    if norm_cauchy >= radius:
        if norm_cauchy == 0:
            return np.zeros_like(p_cauchy), True
        return (radius / norm_cauchy) * p_cauchy, True

    # find tau in [0, 1] with ||p_cauchy + tau * (p_newton - p_cauchy)|| = radius
    d = p_newton - p_cauchy
    a = d.dot(d)
    b = 2 * p_cauchy.dot(d)
    c = p_cauchy.dot(p_cauchy) - radius**2
    tau = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return p_cauchy + tau * d, True


def trust_region_newton(
    fun,
    x0,
    jac="3-point",
    atol=1e-6,
    rtol=1e-6,
    max_iter=100,
    trust_radius0=1.0,
    max_trust_radius=1e10,
    eta=0.1,
):
    """
    Solves f(x) = 0 with a trust-region Newton (Powell dogleg) method.

    Every step of the plain Newton iteration in `newton` is the full
    step J^{-1} f, which can overshoot and diverge far from the
    solution. This solver instead minimizes the merit function
    phi(x) = 0.5 * f(x)^T f(x) using, at each iteration, the best point
    on the dogleg path between the steepest-descent (Cauchy) step and
    the full Newton step that stays inside a trust region of adaptively
    chosen radius. A step is only accepted if the linear model actually
    predicts the observed reduction in phi; otherwise the radius shrinks
    and the step is retried. This makes the iteration robust enough to
    keep converging under much tighter atol/rtol than plain Newton can
    reliably reach.

    Parameters
    ----------
    fun: callable
        Function that takes a vector x and returns a vector f(x), representing
        the system of nonlinear equations.
    x0: array-like
        Initial guess for the solution.
    jac: callable, str, or None, default: "3-point"
        Jacobian function or finite difference approximation method ("2-point",
        "3-point", or "cs").
    atol: float, default: 1e-6
        Absolute tolerance for convergence.
    rtol: float, default: 1e-6
        Relative tolerance for convergence.
    max_iter: int, default: 100
        Maximum number of iterations (accepted and rejected steps combined).
    trust_radius0: float, default: 1.0
        Initial trust-region radius.
    max_trust_radius: float, default: 1e10
        Upper bound the trust-region radius is allowed to grow to.
    eta: float, default: 0.1
        Minimum ratio of actual-to-predicted reduction required to accept
        a step.

    Returns
    -------
    solution: _RichResult
        Container that stores
            - x (array-like): Computed solution.
            - success (bool): Indicates if the method converged.
            - error (float): Final error norm.
            - fun (array-like): Residual function values at the solution.
            - nit (int): Number of iterations performed.
            - nfev (int): Number of function evaluations.
            - njev (int): Number of Jacobian evaluations.
            - rate (float or None): Estimated convergence rate.
            - radius (float): Final trust-region radius.
    """
    nfev = 0
    njev = 0

    # wrap function
    def fun(x, f=fun):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(f(x))

    # wrap jacobian or use a finite difference approximation
    if callable(jac):

        def jacobian(x):
            nonlocal njev
            njev += 1
            return np.atleast_2d(jac(x))

    elif jac in ["2-point", "3-point", "cs"]:

        def jacobian(x):
            nonlocal njev
            njev += 1
            return np.atleast_2d(
                approx_derivative(
                    lambda y: fun(y),
                    x,
                    method=jac,
                    rel_step=1e-6,
                    abs_step=1e-6,
                )
            )

    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    x = x0.copy()

    # initial function value and residual error
    f = fun(x)
    scale_f = atol + np.abs(f) * rtol
    error = np.linalg.norm(f / scale_f) / scale_f.size**0.5
    converged = error < 1

    phi = 0.5 * f.dot(f)
    radius = trust_radius0

    rate = None
    norm_dx_old = None
    i = 0
    if not converged:
        J = jacobian(x)
        for i in range(1, max_iter + 1):
            g = J.T.dot(f)

            # full Newton step solves J p = -f
            try:
                p_newton = linear_solve(J, -f)
            except np.linalg.LinAlgError:
                p_newton = -np.linalg.lstsq(J, f, rcond=None)[0]

            # steepest-descent (Cauchy) step minimizing the linear model of phi
            Jg = J.dot(g)
            Jg_norm2 = Jg.dot(Jg)
            if Jg_norm2 <= 0:
                p_cauchy = np.zeros_like(g)
            else:
                p_cauchy = -(g.dot(g) / Jg_norm2) * g

            p, on_boundary = _dogleg_step(p_newton, p_cauchy, radius)

            # candidate point and actual vs. model-predicted reduction of phi
            x_new = x + p
            f_new = fun(x_new)
            phi_new = 0.5 * f_new.dot(f_new)

            # is the candidate itself already good enough? Once phi is down
            # near the double-precision floor, `actual_reduction` below is a
            # difference of two nearly-equal tiny numbers and is dominated by
            # cancellation noise, making rho meaningless; so tolerance on the
            # candidate point is checked first and, if satisfied, the step is
            # accepted unconditionally rather than risking a spurious
            # rejection (and subsequent stall) from that noise
            scale_x = atol + np.maximum(np.abs(x0), np.abs(x_new)) * rtol
            error_x = np.linalg.norm(p / scale_x) / scale_x.size**0.5
            error_f = np.linalg.norm(f_new / scale_f) / scale_f.size**0.5
            error_new = max(error_f, error_x)
            trial_converged = error_new < 1

            Jp = J.dot(p)
            pred_reduction = -(g.dot(p) + 0.5 * Jp.dot(Jp))
            actual_reduction = phi - phi_new
            rho = actual_reduction / pred_reduction if pred_reduction > 0 else -np.inf

            # trust-region radius update
            if rho < 0.25:
                radius *= 0.25
            elif rho > 0.75 and on_boundary:
                radius = min(2 * radius, max_trust_radius)

            # accept if the model trusts the step, or if it already meets tolerance regardless
            if rho > eta or trial_converged:
                norm_dx = np.linalg.norm(p)
                if norm_dx_old is not None:
                    rate = norm_dx / norm_dx_old
                norm_dx_old = norm_dx

                x, f, phi = x_new, f_new, phi_new
                error = error_new
                converged = trial_converged
                if converged:
                    break
                J = jacobian(x)

            if radius < EPS:
                break

    return _RichResult(
        x=x,
        success=converged,
        error=error,
        fun=f,
        nit=i,
        nfev=nfev,
        njev=njev,
        rate=rate,
        radius=radius,
    )
