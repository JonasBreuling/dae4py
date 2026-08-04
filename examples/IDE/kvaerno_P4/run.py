import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from kvaerno import problem
from dae4py.benchmark import convergence_analysis


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-3
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(2, 2)
    y_true, yp_true = problem.true_sol(t)

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

    # solver options
    h0 = 1e-2
    atol = rtol = 1e-6
    sol = solve_dae_radau(F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol)
    print(sol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(3, 2)
    y_true, yp_true = problem.true_sol(t)

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

    ax[2, 0].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[2, 0].grid()
    ax[2, 0].legend()
    ax[2, 0].set_xscale("log")
    ax[2, 0].set_yscale("log")

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    num = 15
    pow_min = 2
    pow_max = 8
    h_max = Dt / 4
    # h0s = h_max * (1 / 2) ** (np.arange(pow_min, pow_max, dtype=float))
    h0s = h_max * (1 / 2) ** (np.linspace(pow_min, pow_max, num=num, endpoint=True, dtype=float))
    # h0s = h_max * (1 / 2) ** (1 / np.logspace(pow_min, pow_max, num=num, dtype=float))

    atols = 1e-12 * np.ones_like(h0s)
    rtols = 1e-12 * np.ones_like(h0s)

    print(f"rtols: {rtols}")
    print(f"atols: {atols}")
    print(f"h0s: {h0s}")

    errors, rates = convergence_analysis(
        problem,
        rtols,
        atols,
        h0s,
    )


if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    # trajectory(s=2, tableau=radau_tableau)

    # adaptive_radau_IIA(s=3)

    convergence()
