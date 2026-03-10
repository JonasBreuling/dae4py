import numpy as np
from dae4py.consistent_initial_conditions import consistent_initial_conditions

def fun(t, y, yp):
    return np.array(
        [
            2 * yp[0] - y[1],
            y[0] + y[1],
        ]
    )


def jac(t, y, yp):
    Jy = np.array(
        [
            [0, -1],
            [1, 1],
        ]
    )
    Jyp = np.array(
        [
            [2, 0],
            [0, 0],
        ]
    )
    return Jy, Jyp


# initial guess
t0 = 0
y0 = [1, 0]
yp0 = [0, 0]

F0 = fun(t0, y0, yp0)
print(f"- starting values")
print(f"  y0: {y0}")
print(f"  yp0: {yp0}")
print(f"  F0: {F0} => inconsistent")


def naive_algorithm():
    fixed_y0 = []
    fixed_yp0 = []

    y0_consistent, yp0_consistent, F0 = consistent_initial_conditions(
        fun,
        t0,
        y0,
        yp0,
        jac,
        fixed_y0,
        fixed_yp0,
    )
    print(f"- flawed initial conditions")
    print(f"  y0: {y0_consistent}")
    print(f"  yp0: {yp0_consistent}")
    print(f"  F0: {F0} => consistent")


def improved_algorithm():
    fixed_y0 = [0,]
    fixed_yp0 = []

    y0_consistent, yp0_consistent, F0 = consistent_initial_conditions(
        fun,
        t0,
        y0,
        yp0,
        jac,
        fixed_y0,
        fixed_yp0,
    )
    print(f"- desired initial conditions")
    print(f"  y0: {y0_consistent}")
    print(f"  yp0: {yp0_consistent}")
    print(f"  F0: {F0} => consistent")


if __name__ == "__main__":
    naive_algorithm()
    improved_algorithm()