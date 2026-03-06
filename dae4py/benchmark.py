import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau

solvers = [
    ("RadauIIA(1)", solve_dae_IRK, {"tableau": radau_tableau(1)}),
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
    errors = np.zeros((len(solvers), len(rtols), n))
    rates = np.zeros((len(solvers), n))

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
                f=problem.F,
                y0=problem.y0,
                yp0=problem.yp0,
                t_span=problem.t_span,
                h0=h0,
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

            errors[i, j] = error_y

        # estiamte rate of convergence
        log_h = np.vstack([np.log(h0s)] * n).T
        log_err = np.log(errors[i])
        log_err[~np.isfinite(log_err)] = 0

        # estimate slope for h->0
        p = np.diff(log_err, axis=0) / np.diff(log_h, axis=0)
        rates[i] = p[-1]

    for i, name_method_kwargs in enumerate(solvers):
        solver_name = name_method_kwargs[0]
        header = "".join(["h"] + [f", e{i + 1}" for i in range(n)])
        data = np.hstack([h0s[:, None], errors[i]])

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
        for i, ei in enumerate(errors):
            ax[j].plot(
                h0s, ei[:, j], "-o", label=f"{solvers[i][0]}; p≈{rates[i, j]:0.2f}"
            )

        ax[j].set_title(f"convergence analysis:")
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].grid()
        ax[j].legend()
        ax[j].set_xlabel("h [s]")
        ax[j].set_ylabel(f"||y_{j},ref(t1) - y_{j}(t1)||")

    plt.show()

    return errors, rates
