import numpy as np
from dae4py.dae_problem import DAEProblem


def F(t, y, yp):
    y1, y2 = y
    yp1, yp2 = yp
    return np.array(
        [
            (np.sin(yp1) ** 2 + np.cos(y2) ** 2) * yp2**2
            - (t - 6) ** 2 * (t - 2) ** 2 * y1 * np.exp(-t),
            (4 - t) * (y2 + y1) ** 3 - 64 * t**2 * np.exp(-t) * y1 * y2,
        ]
    )


def jac(t, y, yp):
    y1, y2 = y
    yp1, yp2 = yp

    Jy = np.array(
        [
            [
                -((t - 6) ** 2) * (t - 2) ** 2 * np.exp(-t),
                -2 * np.cos(y2) * np.sin(y2) * yp2**2,
            ],
            [
                3 * (4 - t) * (y2 + y1) ** 2 - 64 * t**2 * np.exp(-t) * y2,
                3 * (4 - t) * (y2 + y1) ** 2 - 64 * t**2 * np.exp(-t) * y1,
            ],
        ]
    )

    Jyp = np.array(
        [
            [
                2 * np.sin(yp1) * np.cos(yp1) * yp2**2,
                (np.sin(yp1) ** 2 + np.cos(y2) ** 2) * 2 * yp2,
            ],
            [0, 0],
        ]
    )

    return Jyp, Jy


def true_sol(t):
    return (
        np.array(
            [
                t**4 * np.exp(-t),
                (4 - t) * t**3 * np.exp(-t),
            ]
        ),
        np.array(
            [
                (4 * t**3 - t**4) * np.exp(-t),
                ((4 - t) * 3 * t**2 - (5 - t) * t**3) * np.exp(-t),
            ]
        ),
    )


problem = DAEProblem(
    name="Kvaerno",
    F=F,
    # t_span=(0.1, 1.2),
    # t_span=(0.1, 0.8),
    t_span=(0.5, 1.0), # original values of Kvaerno P4
    index=1,
    true_sol=true_sol,
    jac=jac,
)
