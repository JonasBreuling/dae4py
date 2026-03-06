import numpy as np
from dae4py.dae_problem import DAEProblem

la = -20
phi = lambda t: np.arctan(2 * t)
phi_dot = lambda t: 2 / (1 + 4 * t**2)


def F(t, y, yp):
    return yp - la * (y - phi(t)) - phi_dot(t)


def true_sol(t):
    y = phi(t)
    yp = phi_dot(t)
    return np.atleast_1d(y), np.atleast_1d(yp)


problem = DAEProblem(
    name="Prothero-Robinson problem",
    F=F,
    t_span=(-1, 1),
    index=0,
    true_sol=true_sol,
)
