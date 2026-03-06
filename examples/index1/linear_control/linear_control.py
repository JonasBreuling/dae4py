import numpy as np
from scipy.linalg import expm
from dae4py.dae_problem import DAEProblem

m = np.pi
T = 2.0

q0 = 0.0
q_dot0 = 0.0
x0 = np.array([q0, q_dot0])

q1 = 5.0
q_dot1 = 0.0
x1 = np.array([q1, q_dot1])

R = 1.0
Q = np.zeros((2, 2))

A = np.array(
    [
        [0, 1],
        [0, 0],
    ],
    dtype=float,
)

w = np.array([0, 1 / m], dtype=float)

H = np.block(
    [
        [A, -np.outer(w / R, w)],
        [Q, -A.T],
    ]
)

C = np.block([np.eye(2), np.zeros((2, 2))])

D = np.block([[C], [C @ expm(H * T)]])

z0 = np.linalg.solve(D, np.hstack([x0, x1]))


def true_sol(t):
    z = expm(H * t) @ z0

    x = z[:2]
    la = z[2:4]
    tau = -(w @ la) / R

    z_dot = H @ z
    x_dot = z_dot[:2]
    la_dot = z_dot[2:4]
    tau_dot = -(w @ la_dot) / R

    return (
        np.concatenate([x, la, [tau]]),
        np.concatenate([x_dot, la_dot, [tau_dot]]),
    )


def F(t, y, yp):
    x = y[:2]
    la = y[2:4]
    tau = y[-1]

    x_dot = yp[:2]
    la_dot = yp[2:4]

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[:2] = x_dot - (A @ x + w * tau)
    F[2:4] = la_dot - (Q @ x - A.T @ la)
    F[-1] = R * tau + w @ la

    return F


y0, yp0 = true_sol(0)
print(f"y0: {y0}")
print(f"yp0: {yp0}")

F0 = F(0, y0, yp0)
print(f"F0: {F0}")


problem = DAEProblem(
    name="Linear control problem",
    F=F,
    t_span=(0, T),
    index=1,
    true_sol=true_sol,
)
