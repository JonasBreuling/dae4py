import numpy as np
from itertools import product
import pytest
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau

stages = [2, 3]

buther_tableaus = [
    lambda s: radau_tableau(s),
    lambda s: gauss_legendre_tableau(s),
]

la = [-1, -10, -100]

parameters_dhalquist = product(buther_tableaus, stages, la)
parameters_prothero_robinson = product(buther_tableaus, stages, la)
parameters_brenan = product(buther_tableaus, stages)


@pytest.mark.parametrize("generate_bucher_tableau, s, la", parameters_dhalquist)
def test_dahlquist(generate_bucher_tableau, s, la):
    t0 = 0
    t1 = 1
    t_span = (t0, t1)

    def f(t, y, yp):
        return yp - la * y

    def y_true(t):
        return np.exp(la * (t - t0))

    def yp_true(t):
        return la * np.exp(la * (t - t0))

    y0 = y_true(t0)
    yp0 = yp_true(t0)

    tableau = generate_bucher_tableau(s)

    h = 1e-3

    sol = solve_dae_IRK(f, y0, yp0, t_span, h, tableau)
    t = sol.t
    y = sol.y

    diff = y[:, 0] - y_true(t)
    error = np.linalg.norm(diff) / len(diff) ** 0.5
    print(f"error: {error}")
    # assert error < 1e-6

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(t, y_true(t), "-k", label="y(t) true")
    # ax.plot(t, y, "--r", label="y(t)")
    # ax.legend()
    # ax.grid()
    # plt.show()


@pytest.mark.parametrize("generate_bucher_tableau, s, la", parameters_prothero_robinson)
def test_prothero_robinson(generate_bucher_tableau, s, la):
    t0 = -1.9
    t1 = 1.5
    t_span = (t0, t1)

    def g(t):
        return np.arctan(2 * t)

    def g_dot(t):
        return 2 / (1 + (2 * t) ** 2)

    def f(t, y, yp):
        return yp - (la * (y - g(t)) + g_dot(t))

    def y_true(t):
        return g(t)

    def yp_true(t):
        return g_dot(t)

    y0 = y_true(t0)
    yp0 = yp_true(t0)

    tableau = generate_bucher_tableau(s)

    h = 1e-2

    sol = solve_dae_IRK(f, y0, yp0, t_span, h, tableau, rtol=1e-12, atol=1e-12)
    t = sol.t
    y = sol.y

    diff = y[:, 0] - y_true(t)
    error = np.linalg.norm(diff) / len(diff) ** 0.5
    assert error < 1e-6

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(t, y_true(t), "-k", label="y(t) true")
    # ax.plot(t, y, "--r", label="y(t)")
    # ax.legend()
    # ax.grid()
    # plt.show()


@pytest.mark.parametrize("generate_bucher_tableau, s", parameters_brenan)
def test_brenan(generate_bucher_tableau, s):
    t0 = 0
    t1 = 2.5
    t_span = (t0, t1)

    def f(t, y, yp):
        y1, y2 = y
        y1p, y2p = yp

        F = np.zeros_like(y, dtype=np.common_type(y, yp))
        F[0] = y1p - t * y2p + y1 - (1 + t) * y2
        F[1] = y2 - np.sin(t)
        return F

    def y_true(t):
        return np.array(
            [
                np.exp(-t) + t * np.sin(t),
                np.sin(t),
            ]
        ).T

    def yp_true(t):
        return np.array(
            [
                -np.exp(-t) + np.sin(t) + t * np.cos(t),
                np.cos(t),
            ]
        ).T

    y0 = y_true(t0)
    yp0 = yp_true(t0)

    tableau = generate_bucher_tableau(s)

    h = 1e-2

    sol = solve_dae_IRK(f, y0, yp0, t_span, h, tableau, rtol=1e-12, atol=1e-12)
    t = sol.t
    y = sol.y

    diff = y - y_true(t)
    error = np.linalg.norm(diff) / len(diff) ** 0.5
    assert error < 1e-5

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(t, y_true(t)[:, 0], "-k", label="y1(t) true")
    # ax.plot(t, y_true(t)[:, 1], "-b", label="y2(t) true")
    # ax.plot(t, y[:, 0], "--r", label="y1(t)")
    # ax.plot(t, y[:, 1], "-.g", label="y2(t)")
    # ax.legend()
    # ax.grid()
    # plt.show()


if __name__ == "__main__":
    for generate_bucher_tableau, s, la in parameters_dhalquist:
        test_dahlquist(generate_bucher_tableau, s, la)
    for generate_bucher_tableau, s, la in parameters_prothero_robinson:
        test_prothero_robinson(generate_bucher_tableau, s, la)
    for generate_bucher_tableau, s in parameters_brenan:
        test_brenan(generate_bucher_tableau, s)
