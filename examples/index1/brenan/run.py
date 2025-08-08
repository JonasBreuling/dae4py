import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.genalpha import solve_dae_genalpha
from dae4py.radau import solve_dae_radau
from dae4py.fortran import dassl, pside, radau, radau5
from brenan import problem


def trajectory(s=None, tableau=None, rho_inf=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-1
    atol = rtol = 1e-6
    if s is None or tableau is None:
        if rho_inf is None:
            sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
        else:
            sol = solve_dae_genalpha(
                F, y0, yp0, t_span, h, rho_inf=rho_inf, atol=atol, rtol=rtol
            )
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    y_true, yp_true = problem.true_sol(t)
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"y1")
    ax[0, 0].plot(t, y_true[0], "rx", label=f"y1 true")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"y2")
    ax[1, 0].plot(t, y_true[1], "rx", label=f"y2 true")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"yp1")
    ax[0, 1].plot(t, yp_true[0], "rx", label=f"yp1 true")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"yp2")
    ax[1, 1].plot(t, yp_true[1], "rx", label=f"yp2 true")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0
    # t_span = (0, 1000)
    t_span = (0, 50)

    # solver options
    t_eval = None
    t_eval = np.linspace(*t_span, num=1000)
    h0 = 1e-3
    atol = 1e-6
    rtol = 1e-6

    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval
    )
    # # sol = pside(F, y0, yp0, t_span, rtol=rtol, atol=atol)
    # sol = dassl(F, y0, yp0, t_span, rtol=rtol, atol=atol, t_eval=t_eval)
    # sol = radau(F, y0, yp0, t_span, rtol=rtol, atol=atol)
    # sol = radau5(F, y0, yp0, t_span, rtol=rtol, atol=atol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # sol = dassl(F, y0, yp0, t_span, rtol=rtol, atol=atol, t_eval=t_eval)
    # t = sol["t"]
    # y = sol["y"]
    # yp = sol["yp"]

    print(sol)

    # compute error
    error = np.linalg.norm(y[-1] - problem.true_sol(t)[0][:, -1])
    print(f"error: {error}")

    # visualization
    y_true, yp_true = problem.true_sol(t_eval)
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(t, y[:, 0], "-ok", label=f"y1")
    ax[0, 0].plot(t_eval, y_true[0], "-b", label=f"y1 true")
    if hasattr(sol, "y_eval"):
        ax[0, 0].plot(sol.t_eval, sol.y_eval[:, 0], "--r", label=f"y1 eval")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-ok", label=f"y2")
    ax[1, 0].plot(t_eval, y_true[1], "-b", label=f"y2 true")
    if hasattr(sol, "y_eval"):
        ax[1, 0].plot(sol.t_eval, sol.y_eval[:, 1], "--r", label=f"y2 eval")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-ok", label=f"yp1")
    ax[0, 1].plot(t_eval, yp_true[0], "-b", label=f"yp1 true")
    if hasattr(sol, "y_eval"):
        ax[0, 1].plot(sol.t_eval, sol.yp_eval[:, 0], "--r", label=f"yp1 eval")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-ok", label=f"yp2")
    ax[1, 1].plot(t_eval, yp_true[1], "-b", label=f"yp2 true")
    if hasattr(sol, "y_eval"):
        ax[1, 1].plot(sol.t_eval, sol.yp_eval[:, 1], "--r", label=f"yp2 eval")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 0].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[2, 0].grid()
    ax[2, 0].legend()
    ax[2, 0].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    trajectory(rho_inf=0.5)  # gen-alpha case
    trajectory()  # BDF case
    trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)

    adaptive_radau_IIA(s=7)
