import numpy as np
from tqdm import tqdm
from scipy._lib._util import _RichResult
from dae4py.math import newton


def solve_dae_genalpha(F, y0, yp0, t_span, h, rho_inf=0.5, atol=1e-6, rtol=1e-6):
    """
    Solves a system of DAEs using the generalized-alpha method.

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
    h: float
        Step-size.
    rho_inf: float, default: 0.5
        Dampiing at infinity, 0 <= rho_inf <= 1.
    atol: float, defaul: 1e-6
        Absolute tolerance for the Newton solver.
    rtol: float, default: 1e-6
        Relative tolerance for the Newton solver.

    Returns
    -------
    solution: _RichResult
        Container that stores
        - t (array-like): Time grid.
        - y (array-like): State at the time grid.
        - yp (array-like): Derivative at the time grid.
        - x (array-like): Auxiliary derivative values at the time grid.
    """
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)

    # parameters
    assert 0 <= rho_inf <= 1
    alpha_f = rho_inf / (rho_inf + 1)
    alpha_m = (3 * rho_inf - 1) / (2 * (rho_inf + 1))
    gamma = 0.5 + alpha_f - alpha_m

    # initial guess for auxiliary derivative
    # TODO: Find second-order guess
    x0 = yp0.copy()

    # initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]
    xs = [x0]

    steps = int(np.ceil((t1 - t0) / h))
    with tqdm(total=steps, desc="Generalized-alpha integration") as pbar:
        while t0 < t1:

            def residual(yp):
                # auxiliary derivative and state value
                x = (alpha_f * yp0 + (1 - alpha_f) * yp - alpha_m * x0) / (1 - alpha_m)
                y = y0 + h * ((1 - gamma) * x0 + gamma * x)

                # residual
                return F(t0 + h, y, yp)

                # ####################
                # # Jansen formulation
                # ####################

                # # note: The formulation does not involve the auxiliary
                # # quantity x, but yields in even larger oszillations for
                # # index2 and index 3 systems.

                # # auxiliary derivative and state value
                # y = y0 + h * ((1 - gamma) * yp0 + gamma * yp)
                # # note: the mixing differs from Jansen's approach since we defined the alpha_m and alpha_f differently
                # yp_m = alpha_m * yp0 + (1 - alpha_m) * yp
                # y_f = alpha_f * y0 + (1 - alpha_f) * y

                # # residual
                # # TODO: The evaluation of t is not clear since Jansen investigated autonomous systems.
                # return F(t0 + h, y_f, yp_m)

            # solve the nonlinear system
            sol = newton(residual, yp0, atol=atol, rtol=rtol)
            if not sol.success:
                raise RuntimeError(
                    f"Newton solver failed at t={t0 + h} with error={sol.error:.2e}"
                )

            # extract the solution for stages
            yp1 = sol.x

            # auxiliary derivative and state value
            x1 = (alpha_f * yp0 + (1 - alpha_f) * yp1 - alpha_m * x0) / (1 - alpha_m)
            y1 = y0 + h * ((1 - gamma) * x0 + gamma * x1)

            # append to solution arrays
            t.append(t0 + h)
            y.append(y1)
            yp.append(yp1)
            xs.append(x1)

            # advance time, update initial values and progress bar
            t0 += h
            y0 = y1.copy()
            yp0 = yp1.copy()
            x0 = x1.copy()
            pbar.update(1)

    return _RichResult(
        t=np.array(t),
        y=np.array(y),
        yp=np.array(yp),
        x=np.array(xs),
    )
