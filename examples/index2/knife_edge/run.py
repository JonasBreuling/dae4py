import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.genalpha import solve_dae_genalpha
from knife_edge import problem


def trajectory(s=None, tableau=None, rho_inf=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-2
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


if __name__ == "__main__":
    trajectory(rho_inf=0.9)  # gen-alpha case
    trajectory()  # BDF case
    trajectory(s=3, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)
