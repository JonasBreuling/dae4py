import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from dae4py.genalpha import solve_dae_genalpha_adaptive
from van_der_pol import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-4
    atol = rtol = 1e-6
    sol = solve_dae_radau(F, y0, yp0, t_span, h)
    # if s is None or tableau is None:
    #     sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    # else:
    #     sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"y1")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"y2")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"yp1")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"yp2")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()

    plt.show()


def trajectory_adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = None
    t_eval = np.linspace(*t_span, num=100)
    h0 = 1e-3
    atol = 1e-6
    rtol = 1e-6
    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval
    )
    print(sol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"y1")
    ax[0, 0].plot(sol.t_eval, sol.y_eval[:, 0], "--xr", label=f"y1 eval")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"y2")
    ax[1, 0].plot(sol.t_eval, sol.y_eval[:, 1], "--xr", label=f"y2 eval")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"yp1")
    ax[0, 1].plot(sol.t_eval, sol.yp_eval[:, 0], "--xr", label=f"yp1 eval")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"yp2")
    ax[1, 1].plot(sol.t_eval, sol.yp_eval[:, 1], "--xr", label=f"yp2 eval")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 0].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[2, 0].grid()
    ax[2, 0].legend()
    ax[2, 0].set_yscale("log")

    plt.show()


def trajectory_genalpha_adaptive(rho_inf=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = None
    # t_eval = np.linspace(*t_span, num=100)
    h0 = 1e-3
    atol = 1e-3
    rtol = 1e-3
    sol = solve_dae_genalpha_adaptive(
        F, y0, yp0, t_span, h0, rho_inf=rho_inf, atol=atol, rtol=rtol, t_eval=t_eval
    )
    print(sol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"y1")
    if t_eval is not None:
        ax[0, 0].plot(sol.t_eval, sol.y_eval[:, 0], "--xr", label=f"y1 eval")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"y2")
    if t_eval is not None:
        ax[1, 0].plot(sol.t_eval, sol.y_eval[:, 1], "--xr", label=f"y2 eval")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"yp1")
    if t_eval is not None:
        ax[0, 1].plot(sol.t_eval, sol.yp_eval[:, 0], "--xr", label=f"yp1 eval")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"yp2")
    if t_eval is not None:
        ax[1, 1].plot(sol.t_eval, sol.yp_eval[:, 1], "--xr", label=f"yp2 eval")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 0].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[2, 0].grid()
    ax[2, 0].legend()
    ax[2, 0].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    # trajectory(s=2, tableau=radau_tableau)

    trajectory_adaptive_radau_IIA(s=5)
    trajectory_genalpha_adaptive(rho_inf=0.5)
