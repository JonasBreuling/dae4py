import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from dae4py.math import newton
from linear_control import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-2
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # export
    header = "t, q, u, la1, la2, tau, q_dot, u_dot, la1_dot, la2_dot, tau_dot"
    np.savetxt(
        "linear_control.txt",
        np.column_stack((t, y, yp)),
        delimiter=", ",
        header=header,
        comments="",
    )

    print(f"y_T: {y[-1]}")

    # visualization
    y_true, yp_true = map(np.array, zip(*[problem.true_sol(ti) for ti in t]))

    fig, ax = plt.subplots(5, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"q")
    ax[0, 0].plot(t, y_true[:, 0], "rx", label=f"q true")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"u")
    ax[1, 0].plot(t, y_true[:, 1], "rx", label=f"u true")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[2, 0].plot(t, y[:, 2], "-k", label=f"la1")
    ax[2, 0].plot(t, y_true[:, 2], "rx", label=f"la1 true")
    ax[2, 0].grid()
    ax[2, 0].legend()

    ax[3, 0].plot(t, y[:, 3], "-k", label=f"la2")
    ax[3, 0].plot(t, y_true[:, 3], "rx", label=f"la2 true")
    ax[3, 0].grid()
    ax[3, 0].legend()

    ax[4, 0].plot(t, y[:, 4], "-k", label=f"tau")
    ax[4, 0].plot(t, y_true[:, 4], "rx", label=f"tau true")
    ax[4, 0].grid()
    ax[4, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"q_dot")
    ax[0, 1].plot(t, yp_true[:, 0], "rx", label=f"q_dot true")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"u_dot")
    ax[1, 1].plot(t, yp_true[:, 1], "rx", label=f"u_dot true")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 1].plot(t, yp[:, 2], "-k", label=f"la1_dot")
    ax[2, 1].plot(t, yp_true[:, 2], "rx", label=f"la1_dot true")
    ax[2, 1].grid()
    ax[2, 1].legend()

    ax[3, 1].plot(t, yp[:, 3], "-k", label=f"la2_dot")
    ax[3, 1].plot(t, yp_true[:, 3], "rx", label=f"la2_dot true")
    ax[3, 1].grid()
    ax[3, 1].legend()

    ax[4, 1].plot(t, yp[:, 4], "-k", label=f"tau_dot")
    ax[4, 1].plot(t, yp_true[:, 4], "rx", label=f"tau_dot true")
    ax[4, 1].grid()
    ax[4, 1].legend()

    plt.show()


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = None
    t_eval = np.linspace(*t_span, num=1000)
    h0 = 1e-3
    atol = 1e-6
    rtol = 1e-6

    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval
    )
    t = sol.t
    y = sol.y
    yp = sol.yp
    print(sol)

    t = sol.t_eval
    y = sol.y_eval
    yp = sol.yp_eval

    # export
    header = "t, q, u, la1, la2, tau, q_dot, u_dot, la1_dot, la2_dot, tau_dot"
    np.savetxt(
        "linear_control.txt",
        np.column_stack((t, y, yp)),
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    y_true, yp_true = map(np.array, zip(*[problem.true_sol(ti) for ti in t]))

    fig, ax = plt.subplots(5, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"q")
    ax[0, 0].plot(t, y_true[:, 0], "rx", label=f"q true")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"u")
    ax[1, 0].plot(t, y_true[:, 1], "rx", label=f"u true")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[2, 0].plot(t, y[:, 2], "-k", label=f"la1")
    ax[2, 0].plot(t, y_true[:, 2], "rx", label=f"la1 true")
    ax[2, 0].grid()
    ax[2, 0].legend()

    ax[3, 0].plot(t, y[:, 3], "-k", label=f"la2")
    ax[3, 0].plot(t, y_true[:, 3], "rx", label=f"la2 true")
    ax[3, 0].grid()
    ax[3, 0].legend()

    ax[4, 0].plot(t, y[:, 4], "-k", label=f"tau")
    ax[4, 0].plot(t, y_true[:, 4], "rx", label=f"tau true")
    ax[4, 0].grid()
    ax[4, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"q_dot")
    ax[0, 1].plot(t, yp_true[:, 0], "rx", label=f"q_dot true")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"u_dot")
    ax[1, 1].plot(t, yp_true[:, 1], "rx", label=f"u_dot true")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 1].plot(t, yp[:, 2], "-k", label=f"la1_dot")
    ax[2, 1].plot(t, yp_true[:, 2], "rx", label=f"la1_dot true")
    ax[2, 1].grid()
    ax[2, 1].legend()

    ax[3, 1].plot(t, yp[:, 3], "-k", label=f"la2_dot")
    ax[3, 1].plot(t, yp_true[:, 3], "rx", label=f"la2_dot true")
    ax[3, 1].grid()
    ax[3, 1].legend()

    ax[4, 1].plot(t, yp[:, 4], "-k", label=f"tau_dot")
    ax[4, 1].plot(t, yp_true[:, 4], "rx", label=f"tau_dot true")
    ax[4, 1].grid()
    ax[4, 1].legend()

    plt.show()


if __name__ == "__main__":
    trajectory()  # BDF case
    trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)

    # adaptive_radau_IIA(s=5)
