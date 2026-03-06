import numpy as np
from dae4py.dae_problem import DAEProblem

t0 = 0
t1 = 5
la = -1.5


def F(t, y, yp):
    return yp - la * y


def true_sol(t):
    y = np.exp(la * (t - t0))
    yp = la * y
    return np.atleast_1d(y), np.atleast_1d(yp)


problem = DAEProblem(
    name="Dahlquist",
    F=F,
    t_span=(t0, t1),
    index=0,
    true_sol=true_sol,
)
