import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from particle_circular_track import problem
from dae4py.benchmark import convergence_analysis
from dae4py.radau import solve_dae_radau


def trajectory(s=None, tableau=None):
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 5e-3
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(problem.F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(
            problem.F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol
        )
    t = sol.t
    y = sol.y

    # visualization
    y_true, yp_true = problem.true_sol(t)

    n = len(y0)
    fig, ax = plt.subplots(n, 1)
    for i in range(n):
        ax[i].plot(t, y[:, i], "-r", label=f"y{i}")
        ax[i].plot(t, y_true[i], "x", label=f"y{i} true")
        ax[i].grid()
        ax[i].legend()

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    pow_min = 0
    pow_max = 10
    h_max = Dt / 4
    h0s = h_max * (1 / 2) ** (np.arange(pow_min, pow_max, dtype=float))

    # rtols = 1e-16 * np.ones_like(h0s)
    # atols = 1e-16 * np.ones_like(h0s)
    rtols = 1e-14 * np.ones_like(h0s)
    atols = 1e-14 * np.ones_like(h0s)

    print(f"rtols: {rtols}")
    print(f"atols: {atols}")
    print(f"h0s: {h0s}")

    errors, rates = convergence_analysis(
        problem,
        rtols,
        atols,
        h0s,
    )


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = None
    t_eval = np.linspace(*t_span, num=500)
    h0 = 5e-3
    atol = 1e-8
    rtol = 1e-8

    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval
    )
    t = sol.t
    y = sol.y
    yp = sol.yp
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
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)

    # adaptive_radau_IIA(s=3)

    # convergence()
