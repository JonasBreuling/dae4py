import numpy as np
from dae4py.dae_problem import DAEProblem


A = 1
R = 0.1
B = 0.1
Omega = 0.02 * np.pi
q1 = lambda t: B * np.sin(Omega * t)**2
q1_dot = lambda t: B * Omega * np.sin(2 * Omega * t)

def F(t, y, yp):
    h, q2 = y
    hp, q2p = yp

    F = np.zeros(2, dtype=np.common_type(y, yp))
    F[0] = A * hp - (q1(t) - q2)
    F[1] = q2 - R * np.sqrt(h)

    return F


t0 = 0
t1 = 500
h0 = 2.5
q20 = R * np.sqrt(h0)
y0 = np.array([h0, q20], dtype=float)
yp0 = np.array([(q1(t0) - q20) / A, 0.0], dtype=float)
F0 = F(t0, y0, yp0)
print(f"F0: {F0}")

problem = DAEProblem(
    name="Water tank",
    F=F,
    t_span=(t0, t1),
    index=1,
    y0=y0,
    yp0=yp0,
)
