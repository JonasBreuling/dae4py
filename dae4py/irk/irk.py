import numpy as np
from tqdm import tqdm
from scipy._lib._util import _RichResult
from dae4py.math import newton


def solve_dae_IRK(F, y0, yp0, t_span, h, tableau, jac=None, atol=1e-6, rtol=1e-6):
    """
    Solves a system of DAEs using implicit Runge-Kutta methods.

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
    tableau:
        Butcher tableau defining the IRK method.
    atol: float, defaul: 1e-6
        Absolute tolerance for the Newton solver.
    rtol: float, default: 1e-6
        Relative tolerance for the Newton solver.
    jac: callable
        Jacobian of the method. Returns (M, J) = (dF/dy', dF/dy).

    Returns
    -------
    solution: _RichResult
        Container that stores
        - t (array-like): Time grid.
        - y (array-like): State at the time grid.
        - yp (array-like): Derivative at the time grid.
        - Y (array-like): Stage values at the time grid.
        - Yp (array-like): Stage derivative at the time grid.
    """
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    A, b, c, s = tableau.A, tableau.b, tableau.c, tableau.s

    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
    m = len(y0)
    s = len(c)

    # initial guess for stage derivatives
    Yp = np.tile(yp0, s).reshape(s, -1)
    Y = y0 + h * A.dot(Yp)

    # initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]
    Ys = [Y]
    Yps = [Yp]

    steps = int(np.ceil((t1 - t0) / h))
    with tqdm(total=steps, desc="IRK integration") as pbar:
        while t0 < t1:
            # precompute stage times
            T = t0 + c * h

            def residual(Yp_flat):
                # reshape flat input to stage derivatives
                Yp = Yp_flat.reshape(s, -1)

                # compute stage solutions
                Y = y0 + h * A.dot(Yp)

                # residuals for all stages
                FF = np.zeros((s, m))
                for i in range(s):
                    FF[i] = F(T[i], Y[i], Yp[i])
                return FF.flatten()
            
            if jac is not None:
                def jacobian(Yp_flat):
                    # reshape flat input to stage derivatives
                    Yp = Yp_flat.reshape(s, -1)

                    # compute stage solutions
                    Y = y0 + h * A.dot(Yp)

                    # global Newton matrix
                    J = np.zeros((s * m, s * m))
                    for i in range(s):
                        Jyp_i, Jy_i = jac(T[i], Y[i], Yp[i])

                        row = slice(i * m, (i + 1) * m)
                        for j in range(s):
                            col = slice(j * m, (j + 1) * m)
                            block = h * A[i, j] * Jy_i
                            if i == j:
                                block = block + Jyp_i

                            J[row, col] = block

                    return J
            else:
                jacobian = "3-point"

            # solve the nonlinear system
            sol = newton(residual, Yp.flatten(), jac=jacobian, atol=atol, rtol=rtol)
            if not sol.success:
                raise RuntimeError(
                    f"Newton solver failed at t={t0 + h} with error={sol.error:.2e}"
                )

            # extract the solution for stages
            Yp = sol.x.reshape(s, -1)
            Y = y0 + h * A.dot(Yp)

            # update y and y'
            y1 = y0 + h * b.dot(Yp)
            yp1 = Yp[-1]  # only correct for stiffly accurate methods

            # append to solution arrays
            t.append(t0 + h)
            y.append(y1)
            yp.append(yp1)
            Ys.append(Y)
            Yps.append(Yp)

            # advance time, update initial values and progress bar
            t0 += h
            y0 = y1.copy()
            pbar.update(1)

    return _RichResult(
        t=np.array(t),
        y=np.array(y),
        yp=np.array(yp),
        Y=np.array(Ys),
        Yp=np.array(Yps),
    )
