import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from dae4py.math import newton
from linear_control import problem, x0, x1, z0, H, w, R

if __name__ == "__main__":
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-2
    atol = rtol = 1e-6
    tableau = radau_tableau(2)

    def integrate(la0):
        # consistent initial conditions
        z0 = np.concatenate([x0, la0])
        tau0 = -(w @ la0) / R
        y0 = np.concatenate([x0, la0, [tau0]])

        z_dot0 = H @ z0
        x_dot0 = z_dot0[:2]
        la_dot0 = z_dot0[2:4]
        tau_dot0 = -(w @ la_dot0) / R

        yp0 = np.concatenate([x_dot0, la_dot0, [tau_dot0]])

        # forward simulation
        return solve_dae_IRK(F, y0, yp0, t_span, h, tableau, atol=atol, rtol=rtol)

    def residual(la0):
        sol = integrate(la0)
        xT = sol.y[-1, :2]
        print(f"||x(T) - x_T||: {np.linalg.norm(xT - x1)}")
        return xT - x1

    # 1. naive initial guess
    la0 = np.array([-1, 2], dtype=float)
    # # 2. optimal initial guess => no Newton's method required
    # la0 = z0[2:4]

    # Newton's method
    sol = newton(residual, la0, max_iter=100)
    print(f"Solution: {sol}")

    # extract optimal costate
    la0 = sol.x
    print(f"la0: {la0}")

    # forward integrate the solution with the optimal initial costate
    sol = integrate(la0)
    t = sol.t
    y = sol.y
    yp = sol.yp
    print(f"xT: {sol.y[-1, :2]}; x1: {x1}; error: {np.linalg.norm(sol.y[-1, :2] - x1)}")

    # visualization
    fig, ax = plt.subplots(5, 2)

    ax[0, 0].plot(t, y[:, 0], "-k", label=f"q")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, y[:, 1], "-k", label=f"u")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[2, 0].plot(t, y[:, 2], "-k", label=f"la1")
    ax[2, 0].grid()
    ax[2, 0].legend()

    ax[3, 0].plot(t, y[:, 3], "-k", label=f"la2")
    ax[3, 0].grid()
    ax[3, 0].legend()

    ax[4, 0].plot(t, y[:, 4], "-k", label=f"tau")
    ax[4, 0].grid()
    ax[4, 0].legend()

    ax[0, 1].plot(t, yp[:, 0], "-k", label=f"q_dot")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, yp[:, 1], "-k", label=f"u_dot")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 1].plot(t, yp[:, 2], "-k", label=f"la1_dot")
    ax[2, 1].grid()
    ax[2, 1].legend()

    ax[3, 1].plot(t, yp[:, 3], "-k", label=f"la2_dot")
    ax[3, 1].grid()
    ax[3, 1].legend()

    ax[4, 1].plot(t, yp[:, 4], "-k", label=f"tau_dot")
    ax[4, 1].grid()
    ax[4, 1].legend()

    plt.show()
