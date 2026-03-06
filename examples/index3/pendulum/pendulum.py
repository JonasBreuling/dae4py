import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau

m = 1
l = 1
g = 10


def F(t, vy, vyp, index=3):
    x, y, u, v, la = vy
    x_dot, y_dot, u_dot, v_dot, la_p = vyp

    R = np.zeros(5, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - u
    R[1] = y_dot - v
    R[2] = m * u_dot - (2 * x * la)
    R[3] = m * v_dot - (2 * y * la - m * g)

    match index:
        case 3:
            R[4] = x**2 + y**2 - l**2

        case 2:
            R[4] = 2 * (x * u + y * v)

        case 1:
            R[4] = 2 * (x * u_dot + u**2 + y * v_dot + v**2)

        case 0:
            R[4] = la_p - (
                m / 2 * (v * g - 2 * (u * u_dot + v * v_dot)) / (x**2 + y**2)
                - m
                / 2
                * (y * g - 2 * (u**2 + v**2))
                * (x * u + y * v)
                / (x**2 + y**2) ** 2
            )

    return R


F0 = lambda t, vy, vyp: F(t, vy, vyp, index=0)
F1 = lambda t, vy, vyp: F(t, vy, vyp, index=1)
F2 = lambda t, vy, vyp: F(t, vy, vyp, index=2)
F3 = lambda t, vy, vyp: F(t, vy, vyp, index=3)


def F_GGL(t, vy, vyp):
    x, y, u, v, la, mu = vy
    x_dot, y_dot, u_dot, v_dot, _, _ = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - (u + 2 * x * mu)
    R[1] = y_dot - (v + 2 * y * mu)
    R[2] = m * u_dot - (2 * x * la)
    R[3] = m * v_dot - (2 * y * la - m * g)
    R[4] = 2 * (x * u + y * v)
    R[5] = x**2 + y**2 - 1

    return R


def F_Hiller(t, vy, vyp):
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, kappa_p, nu_p = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - (u + 2 * x * nu_p)
    R[1] = y_dot - (v + 2 * y * nu_p)
    R[2] = m * u_dot - (2 * x * kappa_p)
    R[3] = m * v_dot - (2 * y * kappa_p - m * g)
    R[4] = 2 * (x * u + y * v)
    R[5] = x**2 + y**2 - 1

    return R


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 10
    t_span = (t0, t1)

    # initial conditions
    y0_ = np.array([l, 0, 0, 0, 0], dtype=float)
    yp0_ = np.array([0, 0, 0, -g, 0], dtype=float)
    y0_GGL = np.concatenate([y0_, np.zeros(1)])
    yp0_GGL = np.concatenate([yp0_, np.zeros(1)])

    # solver options
    s = 2
    h = 5e-2

    def solve_DAE(F, y0, yp0):
        return solve_dae_IRK(F, y0, yp0, t_span=t_span, h=h, tableau=radau_tableau(s))

    ##############
    # dae solution
    ##############
    sol0 = solve_DAE(F0, y0_, yp0_)
    t0 = sol0.t
    y0 = sol0.y.T
    yp0 = sol0.yp.T

    sol1 = solve_DAE(F1, y0_, yp0_)
    t1 = sol1.t
    y1 = sol1.y.T
    yp1 = sol1.yp.T

    sol2 = solve_DAE(F2, y0_, yp0_)
    t2 = sol2.t
    y2 = sol2.y.T
    yp2 = sol2.yp.T

    sol3 = solve_DAE(F3, y0_, yp0_)
    t3 = sol3.t
    y3 = sol3.y.T
    yp3 = sol3.yp.T

    sol_GGL = solve_DAE(F_GGL, y0_GGL, yp0_GGL)
    t_GGL = sol_GGL.t
    y_GGL = sol_GGL.y.T
    yp_GGL = sol_GGL.yp.T

    sol_Hiller = solve_DAE(F_Hiller, y0_GGL, yp0_GGL)
    t_Hiller = sol_Hiller.t
    y_Hiller = sol_Hiller.y.T
    yp_Hiller = sol_Hiller.yp.T

    # export solution
    import sys
    from pathlib import Path

    header = "t_pos, g_pos, t_vel, g_vel, t_acc, g_acc, t_ODE, g_ODE, t_GGL, g_GGL, t_Hiller, g_Hiller"

    data = np.vstack(
        (
            t3[None, :],
            (y3[0] ** 2 + y3[1] ** 2 - l**2)[None, :],
            t2[None, :],
            (y2[0] ** 2 + y2[1] ** 2 - l**2)[None, :],
            t1[None, :],
            (y1[0] ** 2 + y1[1] ** 2 - l**2)[None, :],
            t0[None, :],
            (y0[0] ** 2 + y0[1] ** 2 - l**2)[None, :],
            t_GGL[None, :],
            (y_GGL[0] ** 2 + y_GGL[1] ** 2 - l**2)[None, :],
            t_Hiller[None, :],
            (y_Hiller[0] ** 2 + y_Hiller[1] ** 2 - l**2)[None, :],
        )
    ).T

    path = Path(sys.modules["__main__"].__file__)

    np.savetxt(
        path.parent / (path.stem + ".txt"),
        data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # visualization
    fig, ax = plt.subplots(5, 1)

    ax[0].plot(t0, y0[0], "-o", label="x ODE.")
    ax[0].plot(t1, y1[0], "-", label="x acc.")
    ax[0].plot(t2, y2[0], "--", label="x vel.")
    ax[0].plot(t3, y3[0], ":", label="x pos.")
    ax[0].plot(t_GGL, y_GGL[0], "-.", label="x GGL")
    ax[0].plot(t_Hiller, y_Hiller[0], "--x", label="x Hiller")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t0, y0[1], "-o", label="y ODE.")
    ax[1].plot(t1, y1[1], "-", label="y acc.")
    ax[1].plot(t2, y2[1], "--", label="y vel.")
    ax[1].plot(t3, y3[1], ":", label="y pos.")
    ax[1].plot(t_GGL, y_GGL[1], "-.", label="y GGL")
    ax[1].plot(t_Hiller, y_Hiller[1], "--x", label="y Hiller")
    ax[1].legend()
    ax[1].grid()

    g = lambda y: y[0] ** 2 + y[1] ** 2 - l**2
    ax[2].plot(t0, g(y0), "-o", label="g ODE.")
    ax[2].plot(t1, g(y1), "-", label="g acc.")
    ax[2].plot(t2, g(y2), "--", label="g vel.")
    ax[2].plot(t3, g(y3), ":", label="g pos.")
    ax[2].plot(t_GGL, g(y_GGL), "-.", label="g GGL")
    ax[2].plot(t_Hiller, g(y_Hiller), "--x", label="g Hiller")
    ax[2].legend()
    ax[2].grid()
    ax[2].set_yscale("symlog", linthresh=1e-4)

    g_dot = lambda y: 2 * (y[0] * y[2] + y[1] * y[3])
    ax[3].plot(t1, g_dot(y0), "-o", label="g_dot ODE.")
    ax[3].plot(t1, g_dot(y1), "-", label="g_dot acc.")
    ax[3].plot(t2, g_dot(y2), "--", label="g_dot vel.")
    ax[3].plot(t3, g_dot(y3), ":", label="g_dot pos.")
    ax[3].plot(t_GGL, g_dot(y_GGL), "-.", label="g_dot GGL")
    ax[3].plot(t_Hiller, g_dot(y_Hiller), "--x", label="g_dot Hiller")
    ax[3].legend()
    ax[3].grid()
    ax[3].set_yscale("symlog", linthresh=1e-4)

    g_ddot = lambda y, yp: 2 * (y[0] * yp[2] + y[2] ** 2 + y[1] * yp[3] + y[3] ** 2)
    ax[4].plot(t1, g_ddot(y0, yp0), "-o", label="g_ddot ODE.")
    ax[4].plot(t1, g_ddot(y1, yp1), "-", label="g_ddot acc.")
    ax[4].plot(t2, g_ddot(y2, yp2), "--", label="g_ddot vel.")
    ax[4].plot(t3, g_ddot(y3, yp3), ":", label="g_dot pos.")
    ax[4].plot(t_GGL, g_ddot(y_GGL, yp_GGL), "-.", label="g_dot GGL")
    ax[4].plot(t_Hiller, g_ddot(y_Hiller, yp_Hiller), "--x", label="g_dot Hiller")
    ax[4].legend()
    ax[4].grid()
    ax[4].set_yscale("symlog", linthresh=1e-2)

    plt.show()
