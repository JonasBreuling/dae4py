import numpy as np
import matplotlib.pyplot as plt
from dae4py.erk import solve_ode, Heun, RK23, RK45

if __name__ == "__main__":
    eps = 1e-3
    mu = 1 / np.sqrt(eps)

    def rhs(t, y):
        z1, z2 = y
        return np.array(
            [
                z2,
                ((1 - z1**2) * z2 - z1) / eps,
            ]
        )

    t0 = 0
    t1 = 5
    t_span = [t0, t1]
    t_eval = np.linspace(*t_span, num=int(3e2))
    y0 = np.array([2, 0], dtype=float)

    # method = Heun
    method = RK23
    # method = RK45

    sol = solve_ode(rhs, t0, y0, t_span=t_span, t_eval=t_eval)
    t, h, y, y_eval = sol.t, sol.h, sol.y, sol.y_eval

    fig, ax = plt.subplots(3)

    ax[0].plot(t, y[:, 0], "-o", label="y1")
    ax[0].plot(t_eval, y_eval[:, 0], "--x", label="y1 dense")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, y[:, 1], "-o", label="y2")
    ax[1].plot(t_eval, y_eval[:, 1], "--x", label="y2 dense")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, h, "-o", label="h")
    ax[2].grid()
    ax[2].legend()
    ax[2].set_yscale("log")

    plt.show()
