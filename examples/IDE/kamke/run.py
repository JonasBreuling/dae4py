import time
import numpy as np
import matplotlib.pyplot as plt
from dae4py.radau import solve_dae_radau
from kamke import KamkeDAEProblem

if __name__ == "__main__":
    # generate the problem
    C = 1
    problem = KamkeDAEProblem(C)

    # run the solver
    atol = rtol = 1e-6
    start = time.time()
    sol = solve_dae_radau(
        problem.F,
        problem.y0,
        problem.yp0,
        problem.t_span,
        atol=atol,
        rtol=rtol,
        jac=problem.jac,
    )
    end = time.time()
    t = sol.t
    y = sol.y.T

    # error
    error = np.linalg.norm(y[:, -1] - problem.true_sol(t[-1])[0])
    print(f"error: {error}")

    # visualization
    fig, ax = plt.subplots()

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.plot(t, problem.true_sol(t)[0], "-ok", label="y_true")
    ax.plot(t, y[0], "--xr", label=f"y")

    ax.grid()
    ax.legend()
    plt.show()
