import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from knife_edge import problem
from dae4py.benchmark import convergence_analysis


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

    # visualization
    y_true, yp_true = problem.true_sol(t)
    n = len(y0)
    fig, ax = plt.subplots(n, 2)

    for i in range(n):
        ax[i, 0].plot(t, y[:, i], "-k", label=f"y{i+1}")
        ax[i, 0].plot(t, y_true[i], "rx", label=f"y{i+1} true")
        ax[i, 0].grid()
        ax[i, 0].legend()

        ax[i, 1].plot(t, yp[:, i], "-k", label=f"yp{i+1}")
        ax[i, 1].plot(t, yp_true[i], "rx", label=f"yp{i+1} true")
        ax[i, 1].grid()
        ax[i, 1].legend()

    plt.show()


def convergence():
    Dt = problem.t1 - problem.t0
    pow_min = 0
    pow_max = 14
    h_max = Dt / 4
    h0s = h_max * (1 / 2) ** (np.arange(pow_min, pow_max, dtype=float))

    # rtols = 1e-16 * np.ones_like(h0s)
    # atols = 1e-16 * np.ones_like(h0s)
    rtols = 1e-17 * np.ones_like(h0s)
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


if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=3, tableau=gauss_legendre_tableau)
    # trajectory(s=2, tableau=radau_tableau)

    convergence()
