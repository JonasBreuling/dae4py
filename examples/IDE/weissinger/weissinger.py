import numpy as np
from dae4py.dae_problem import DAEProblem


def F(t, y, yp):
    return t * y**2 * yp**3 - y**3 * yp**2 + t * (t**2 + 1) * yp - t**2 * y


def jac(t, y, yp):
    Jyp = np.array([3 * t * y**2 * yp**2 - 2 * y**3 * yp + t * (t**2 + 1)])
    Jy = np.array(
        [
            2 * t * y * yp**3 - 3 * y**2 * yp**2 - t**2 * y,
        ]
    )
    return Jyp, Jy


def true_sol(t):
    return np.atleast_1d(np.sqrt(t**2 + 0.5)), np.atleast_1d(t / np.sqrt(t**2 + 0.5))


problem = DAEProblem(
    name="Weissinger",
    F=F,
    # t_span=(np.sqrt(0.5), 10),
    t_span=(np.sqrt(0.5), 3),
    index=0,
    true_sol=true_sol,
    jac=jac,
)
