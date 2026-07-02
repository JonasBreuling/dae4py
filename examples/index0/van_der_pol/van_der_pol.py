import numpy as np
from dae4py.dae_problem import DAEProblem

eps = 1e-2
t0 = 0
t1 = 550


def rhs(t, y):
    z1, z2 = y
    return np.array(
        [
            z2,
            (1 - z1**2) * z2 / eps - z1,
        ]
    )


def F(t, y, yp):
    # return yp - rhs(t, y)  # naive ODE call
    z1, z2 = y
    zp1, zp2 = yp
    return np.array([zp1 - z2, eps * zp2 - (1 - z1**2) * z2 + eps * z1])


y0 = np.array([2, 0], dtype=float)
yp0 = rhs(t0, y0)


problem = DAEProblem(
    name="Van der Pol oscillator",
    F=F,
    t_span=(t0, t1),
    index=0,
    y0=y0,
    yp0=yp0,
)
