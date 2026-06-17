import numpy as np
from scipy._lib._util import _RichResult
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import lu_factor, lu_solve


def newton(
    fun,
    x0,
    jac="3-point",
    atol=1e-6,
    rtol=1e-6,
    max_iter=100,
    chord=False,
):
    """
    This function implements the Newton-Raphson method for solving nonlinear
    equations of the form f(x) = 0. It supports both the standard Newton method
    and the chord method, where the Jacobian is computed once and reused in
    subsequent iterations.

    Parameters
    ----------
    fun: callable
        Function that takes a vector x and returns a vector f(x), representing
        the system of nonlinear equations.
    x0: array-like
        Initial guess for the solution.
    jac: callable, str, or None, default: "2-point"
        Jacobian function or finite difference approximation method ("2-point",
        "3-point", or "cs").
    atol: float, default: 1e-6
        Absolute tolerance for convergence.
    rtol: float, default: 1e-6
        Relative tolerance for convergence.
    max_iter: int, default: 20
        Maximum number of iterations.
    chord: bool, default: True
        If True, uses the chord method by computing the Jacobian once.

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
            return jac(x)

    elif jac in ["2-point", "3-point", "cs"]:

        def jacobian(x):
            nonlocal njev
            njev += 1
            return approx_derivative(
                lambda y: fun(y),
                x,
                method=jac,
                rel_step=1e-6,
                abs_step=1e-6,
            )

    # eliminate round-off errors
    Delta_x = np.zeros_like(x0)
    x = x0 + Delta_x

    # initial function value
    f = fun(x)

    # scaling with relative and absolute tolerances
    scale_f = atol + np.abs(f) * rtol

    # error of initial guess
    error = np.linalg.norm(f / scale_f) / scale_f.size**0.5
    converged = error < 1

    # Newton loop
    norm_dx_old = 1
    rate = None
    i = 0
    LU = None
    if not converged:
        for i in range(1, max_iter + 1):
            # evaluate Jacobian
            if chord and LU is None:
                J = np.atleast_2d(jacobian(x))
                LU = lu_factor(J)

            # Newton update
            if chord:
                dx = lu_solve(LU, f)
            else:
                J = np.atleast_2d(jacobian(x))
                dx = np.linalg.solve(J, f)

            # estimate rate of convergence
            norm_dx = np.linalg.norm(dx)
            if i > 1:
                rate = norm_dx / norm_dx_old
            norm_dx_old = norm_dx

            # perform Newton step
            Delta_x -= dx
            x = x0 + Delta_x

            # new function value and residual error
            f = np.atleast_1d(fun(x))
            error_f = np.linalg.norm(f / scale_f) / scale_f.size**0.5

            # step-size error
            scale_x = atol + np.maximum(np.abs(x0), np.abs(x)) * rtol
            error_x = np.linalg.norm(dx / scale_x) / scale_x.size**0.5

            error = max(error_f, error_x)
            converged = error < 1
            if converged:
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
    )
