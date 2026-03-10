import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from akzo_nobel import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-2
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # export
    header = "t, y1, y2, y3, y4, y5, z, y_dot1, y_dot2, y_dot3, y_dot4, y_dot5, z_dot"
    np.savetxt(
        "akzo_nobel.txt",
        np.column_stack((t, y, yp)),
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    n = len(y0)
    fig, ax = plt.subplots(n, 2)

    for i in range(n):
        ax[i, 0].plot(t, y[:, i], "-", label=f"y{i+1}")
        ax[i, 1].plot(t, yp[:, i], "-", label=f"yp{i+1}")

        ax[i, 0].grid()
        ax[i, 0].set_xscale("log")
        ax[i, 0].legend()
        
        ax[i, 1].grid()
        ax[i, 1].set_xscale("log")
        ax[i, 1].legend()

    plt.show()


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = None
    # t_eval = np.linspace(*t_span, num=100)
    t_eval = np.logspace(*t_span, num=100)
    t_eval = np.logspace(-3, 2, num=500)
    h0 = 1e-3
    atol = 1e-6
    rtol = 1e-6

    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval, max_step=1e-1
    )
    t = sol.t
    y = sol.y
    yp = sol.yp
    print(sol)

    t = sol.t_eval
    y = sol.y_eval
    yp = sol.yp_eval

    # export
    header = "t, y1, y2, y3, y4, y5, z, y_dot1, y_dot2, y_dot3, y_dot4, y_dot5, z_dot"
    np.savetxt(
        "akzo_nobel.txt",
        np.column_stack((t, y, yp)),
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    n = len(y0)
    fig, ax = plt.subplots(n, 2)

    for i in range(n):
        ax[i, 0].plot(t, y[:, i], "-", label=f"y{i+1}")
        ax[i, 1].plot(t, yp[:, i], "-", label=f"yp{i+1}")

        ax[i, 0].grid()
        ax[i, 0].set_xscale("log")
        ax[i, 0].legend()
        
        ax[i, 1].grid()
        ax[i, 1].set_xscale("log")
        ax[i, 1].legend()

    plt.show()

if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    # trajectory(s=3, tableau=radau_tableau)

    adaptive_radau_IIA(s=3)
