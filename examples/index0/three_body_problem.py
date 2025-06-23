import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dae4py.radau import solve_dae_radau

# gravitational constant (normalized)
G = 1.0

# equal masses for all three bodies
m1 = m2 = m3 = 1.0
# m1 += 5e-2  # small mass pertubation
ms = [m1, m2, m3]

# figure-8 initial conditions (from known approximations)
q1 = np.array([0.97000436, -0.24308753])
q2 = -q1.copy()
q3 = np.zeros(2)
q0 = np.concatenate((q1, q2, q3))

u1 = np.array([0.466203685, 0.43236573])
u2 = u1.copy()
u3 = -2 * u1.copy()
u0 = np.concatenate((u1, u2, u3))

# export initial conditions
np.savetxt(
    "three_body_problem_initial_conditions.txt",
    np.hstack(
        [
            np.vstack([q1, q2, q3]),
            np.vstack([u1, u2, u3]),
        ]
    ),
    header="x, y, u, v",
    delimiter=", ",
    comments="",
)


def rhs(t, y):
    q, u = y[:6], y[6:]
    q1, q2, q3 = q[:2], q[2:4], q[4:6]
    qs = q.reshape(-1, 2)

    q_dot = u
    u_dot = -np.concatenate(
        [
            sum(ms[j] * (qs[0] - qs[j]) / norm((q1 - qs[j])) ** 3 for j in [1, 2]),
            sum(ms[j] * (qs[1] - qs[j]) / norm((q2 - qs[j])) ** 3 for j in [0, 2]),
            sum(ms[j] * (qs[2] - qs[j]) / norm((q3 - qs[j])) ** 3 for j in [0, 1]),
        ]
    )

    return np.concatenate((q_dot, u_dot))


def F(t, y, yp):
    return yp - rhs(t, y)


# time span for the simulation
t_span = (0, 6.4)
t_eval = np.linspace(*t_span, 2000)

# initial conditions
y0 = np.concatenate((q0, u0))
yp0 = rhs(t_span[0], y0)

# solve the system of ODEs
sol = solve_dae_radau(F, y0, yp0, t_span, t_eval=t_eval, atol=1e-6, rtol=1e-6, s=5)

# extract positions
q = sol.y_eval[:, :6]
x1, y1, x2, y2, x3, y3 = q.T

# export numerical solution
np.savetxt(
    "three_body_problem_solution.txt",
    q,
    header="x1, y1, x2, y2, x3, y3",
    delimiter=", ",
    comments="",
)

# plot the trajectories
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, label="Body 1 (m1)", lw=2)
plt.plot(x2, y2, label="Body 2 (m2)", lw=2)
plt.plot(x3, y3, label="Body 3 (m3)", lw=2)
plt.scatter(
    y0[0::2][:3],
    y0[1::2][:3],
    c=["blue", "orange", "green"],
    s=100,
    label="Initial positions",
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure-8 Orbit in the Three-Body Problem")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

# animate the figure-8 orbit
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Figure-8 Orbit Animation")

# initialize trajectories and points
(line1,) = ax.plot([], [], "b-", label="Body 1 (m1)")
(line2,) = ax.plot([], [], "r-", label="Body 2 (m2)")
(line3,) = ax.plot([], [], "g-", label="Body 3 (m3)")
(point1,) = ax.plot([], [], "bo", markersize=int(12 * m1))
(point2,) = ax.plot([], [], "ro", markersize=int(12 * m2))
(point3,) = ax.plot([], [], "go", markersize=int(12 * m3))

# legend and grid
ax.legend()
ax.grid()


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    return line1, line2, line3, point1, point2, point3


def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    point1.set_data(x1[frame], y1[frame])
    point2.set_data(x2[frame], y2[frame])
    point3.set_data(x3[frame], y3[frame])
    return line1, line2, line3, point1, point2, point3


frames = len(sol.t_eval)
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=10)

plt.show()
