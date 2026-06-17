import numpy as np
from dae4py.dae_problem import DAEProblem


def rhs(t, y):
    y1, y2, y3 = y

    yp = np.zeros(3, dtype=y.dtype)
    yp[0] = -0.04 * y1 + 1e4 * y2 * y3
    yp[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    yp[2] = 3e7 * y2**2

    return yp


def F(t, y, yp):
    return yp - rhs(t, y)
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=np.common_type(y, yp))
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F


t0 = 1e-10
t1 = 5e1
t1 = 1e6
y0 = np.array([1, 0, 0], dtype=float)
yp0 = rhs(t0, y0)

problem = DAEProblem(
    name="Robertson",
    F=F,
    t_span=(t0, t1),
    index=1,
    y0=y0,
    yp0=yp0,
)
