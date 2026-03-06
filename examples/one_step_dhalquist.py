import numpy as np
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau

la = 1.5


def F(t, y, yp):
    return yp - la * y


def true_sol(t):
    return np.exp(la * t), la * np.exp(la * t)


if __name__ == "__main__":
    # time span
    t0 = 0.0
    t1 = 1.0
    t_span = (t0, t1)

    # initial conditions
    y0, yp0 = true_sol(t0)

    # solver options
    h = 1.0
    atol = rtol = 1e-14

    # Radau method
    s = 2
    atol = rtol = 1e-14

    # single step of RadauIIA(2)
    tableau = radau_tableau(s)
    sol_RadauIIA2 = solve_dae_IRK(
        F, y0, yp0, t_span, h, tableau=tableau, atol=atol, rtol=rtol
    )

    np.savetxt(
        "RadauIIA(2)_dahlquist.txt",
        np.array(
            [
                [t0, t1],
                [y0, sol_RadauIIA2.y[-1][0]],
                [t0, t0 + tableau.c[0] * h],
                [t0, t0 + tableau.c[1] * h],
                [y0, sol_RadauIIA2.Y[-1][0][0]],
                [y0, sol_RadauIIA2.Y[-1][1][0]],
                [yp0, sol_RadauIIA2.Yp[-1][0][0]],
                [yp0, sol_RadauIIA2.Yp[-1][1][0]],
                [h, h],
                [la, la],
            ]
        ).T,
        header="t, y, tau1, tau2, Y1, Y2, Y_dot1, Y_dot2, h, la",
        delimiter=", ",
        comments="",
    )

    # single step of GaussLegendre(2)
    tableau = gauss_legendre_tableau(s)
    sol_GaussLegendre2 = solve_dae_IRK(
        F, y0, yp0, t_span, h, tableau=tableau, atol=atol, rtol=rtol
    )

    np.savetxt(
        "GaussLegendre(2)_dahlquist.txt",
        np.array(
            [
                [t0, t1],
                [y0, sol_GaussLegendre2.y[-1][0]],
                [t0, t0 + tableau.c[0] * h],
                [t0, t0 + tableau.c[1] * h],
                [y0, sol_GaussLegendre2.Y[-1][0][0]],
                [y0, sol_GaussLegendre2.Y[-1][1][0]],
                [yp0, sol_GaussLegendre2.Yp[-1][0][0]],
                [yp0, sol_GaussLegendre2.Yp[-1][1][0]],
                [h, h],
                [la, la],
            ]
        ).T,
        header="t, y, tau1, tau2, Y1, Y2, Y_dot1, Y_dot2, h, la",
        delimiter=", ",
        comments="",
    )
