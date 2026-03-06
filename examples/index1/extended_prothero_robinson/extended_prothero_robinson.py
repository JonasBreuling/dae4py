import numpy as np
from dae4py.dae_problem import DAEProblem

omega = 3
eps = 1e3
la = -20

phi1 = lambda t: np.arctan(eps * np.cos(omega * t))
phi2 = lambda t: np.sin(t)

phi1_dot = (
    lambda t: -eps * omega * np.sin(omega * t) / (1 + (eps * np.cos(omega * t)) ** 2)
)
phi2_dot = lambda t: np.cos(t)


def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - la * (y1 - phi1(t) * y2) - phi1_dot(t) * y2 - phi1(t) * y2p
    F[1] = y2 - phi2(t)

    return F


def true_sol(t):
    return (
        np.array(
            [
                phi1(t) * phi2(t),
                phi2(t),
            ]
        ),
        np.array(
            [
                phi1_dot(t) * phi2(t) + phi1(t) * phi2_dot(t),
                phi2_dot(t),
            ]
        ),
    )


problem = DAEProblem(
    name="Extended Prothero-Robinson problem",
    F=F,
    t_span=(0, 10),
    index=1,
    true_sol=true_sol,
)
