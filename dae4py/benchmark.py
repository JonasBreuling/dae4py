import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau

solvers = [
    # ("RadauIIA(1)", solve_dae_IRK, {"tableau": radau_tableau(1)}),
    ("RadauIIA(2)", solve_dae_IRK, {"tableau": radau_tableau(2)}),
    ("RadauIIA(3)", solve_dae_IRK, {"tableau": radau_tableau(3)}),
    # # ("Gauss-Legendre(1)", solve_dae_IRK, {"tableau": gauss_legendre_tableau(1)}),
    # # ("Gauss-Legendre(2)", solve_dae_IRK, {"tableau": gauss_legendre_tableau(2)}),
    # ("Gauss-Legendre(3)", solve_dae_IRK, {"tableau": gauss_legendre_tableau(3)}),
    # ("Gauss-Legendre(4)", solve_dae_IRK, {"tableau": gauss_legendre_tableau(4)}),
    # ("Gauss-Legendre(5)", solve_dae_IRK, {"tableau": gauss_legendre_tableau(5)}),
]


def convergence_analysis(problem, rtols, atols, h0s):
    # benchmark results
    n = len(problem.y0)
    errors_y = np.zeros((len(solvers), len(rtols), n))
    errors_yp = np.zeros((len(solvers), len(rtols), n))
    rates_y = np.zeros((len(solvers), n))
    rates_yp = np.zeros((len(solvers), n))

    for i, name_method_kwargs in enumerate(solvers):
        solver_name, method, kwargs = name_method_kwargs
        print(f" - method: {solver_name}; kwargs: {kwargs}")
        for j, (rtol, atol, h0) in enumerate(zip(rtols, atols, h0s)):
            print(f"   * rtol: {rtol}")
            print(f"   * atol: {atol}")
            print(f"   * h0:   {h0}")

            # solve system
            start = time.time()
            sol = method(
                F=problem.F,
                y0=problem.y0,
                yp0=problem.yp0,
                t_span=problem.t_span,
                h=h0,
                atol=atol,
                rtol=rtol,
                **kwargs,
            )
            end = time.time()
            elapsed_time = end - start

            # error
            y_true, yp_true = problem.true_sol(problem.t1)
            idx = np.where(np.isclose(sol.t, problem.t1))[0][0]
            diff_y = y_true - sol.y[idx]
            error_y = np.abs(diff_y)
            print(f"     => error_y: {error_y}")
            diff_yp = yp_true - sol.yp[idx]
            error_yp = np.abs(diff_yp)
            print(f"     => error_yp: {error_yp}")

            errors_y[i, j] = error_y
            errors_yp[i, j] = error_yp

        # estiamte rate of convergence
        log_h = np.vstack([np.log(h0s)] * n).T
        log_err_y = np.log(errors_y[i])
        log_err_y[~np.isfinite(log_err_y)] = 0

        # estimate slope for h->0
        p_y = np.diff(log_err_y, axis=0) / np.diff(log_h, axis=0)
        rates_y[i] = p_y[-1]

        # estiamte rate of convergence
        log_err_yp = np.log(errors_yp[i])
        log_err_yp[~np.isfinite(log_err_yp)] = 0

        # estimate slope for h->0
        p_yp = np.diff(log_err_yp, axis=0) / np.diff(log_h, axis=0)
        rates_yp[i] = p_yp[-1]

    for i, name_method_kwargs in enumerate(solvers):
        solver_name = name_method_kwargs[0]
        header = "".join(["h"] + [f", e{i + 1}" for i in range(n)])
        data = np.hstack([h0s[:, None], errors_y[i]])

        np.savetxt(
            f"{problem.name}_index{problem.index}_{solver_name}_convergence.txt",
            data,
            header=header,
            delimiter=", ",
            comments="",
        )

    fig, ax = plt.subplots(1, n, figsize=(12, 9))

    for j in range(n):
        ax[j].plot(h0s, h0s, "--", label="h")
        ax[j].plot(h0s, h0s**2, "--", label="h^2")
        ax[j].plot(h0s, h0s**3, "--", label="h^3")
        ax[j].plot(h0s, h0s**4, "--", label="h^4")
        ax[j].plot(h0s, h0s**5, "--", label="h^5")
        ax[j].plot(h0s, h0s**6, "--", label="h^6")
        for i, ei in enumerate(errors_y):
            ax[j].plot(
                h0s, ei[:, j], "-o", label=f"{solvers[i][0]}; p≈{rates_y[i, j]:0.2f}"
            )

        ax[j].set_title(f"convergence analysis:")
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].grid()
        ax[j].legend()
        ax[j].set_xlabel("h [s]")
        ax[j].set_ylabel(f"||y_{j},ref(t1) - y_{j}(t1)||")

    fig, ax = plt.subplots(1, n, figsize=(12, 9))

    for j in range(n):
        ax[j].plot(h0s, h0s, "--", label="h")
        ax[j].plot(h0s, h0s**2, "--", label="h^2")
        ax[j].plot(h0s, h0s**3, "--", label="h^3")
        ax[j].plot(h0s, h0s**4, "--", label="h^4")
        ax[j].plot(h0s, h0s**5, "--", label="h^5")
        ax[j].plot(h0s, h0s**6, "--", label="h^6")
        for i, ei in enumerate(errors_yp):
            ax[j].plot(
                h0s, ei[:, j], "-o", label=f"{solvers[i][0]}; p≈{rates_yp[i, j]:0.2f}"
            )

        ax[j].set_title(f"convergence analysis:")
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].grid()
        ax[j].legend()
        ax[j].set_xlabel("h [s]")
        ax[j].set_ylabel(f"||yp_{j},ref(t1) - yp_{j}(t1)||")

    plt.show()

    return errors_y, rates_y
