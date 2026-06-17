import numpy as np
from scipy._lib._util import _RichResult

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step-size.
MAX_FACTOR = 10  # Maximum allowed increase in a step-size.


class ExplicitButcherTableau:
    A = NotImplemented
    b = NotImplemented
    b_hat = NotImplemented
    c = NotImplemented

    n_stages = NotImplemented
    order = NotImplemented
    error_estimator_order = NotImplemented

    @classmethod
    def verify_tableau(cls):
        assert np.allclose(cls.A[0], np.zeros_like(cls.A[0])), f"Method '{cls}' is not FSAL"
        assert np.allclose(cls.A, np.tril(cls.A, -1)), f"Method '{cls}' is implicit"
        assert cls.A.shape == (cls.n_stages, cls.n_stages), f"Method '{cls}' has flawed coefficient matrix A"
        assert len(cls.b) == cls.n_stages, f"Method '{cls}' has flawed quadrature weights b"
        assert len(cls.c) == cls.n_stages, f"Method '{cls}' has flawed quadrature weights b_hat"
        assert len(cls.b_hat) == cls.n_stages + 1, f"Method '{cls}' has flawed quadrature weights b_hat"
        assert cls.order != cls.error_estimator_order, f"Method '{cls}' has same order for embedded method"


class Heun(ExplicitButcherTableau):
    # fmt: off
    A = np.array([
        [0, 0],
        [1, 0],
    ], dtype=float)
    # fmt: on
    b = np.array([1 / 2, 1 / 2])
    b_hat = np.array([1, 0, 0], dtype=float)
    c = np.array([0, 1], dtype=float)

    n_stages = 2
    order = 2
    error_estimator_order = 1


class RK23(ExplicitButcherTableau):
    # fmt: off
    A = np.array([
        [  0,    0, 0],
        [1/2,    0, 0],
        [   0, 3/4, 0]
    ])
    # fmt: on
    b = np.array([2 / 9, 1 / 3, 4 / 9])
    b_hat = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
    c = np.array([0, 1 / 2, 3 / 4])

    n_stages = 3
    order = 3
    error_estimator_order = 2


class RK45(ExplicitButcherTableau):
    # fmt: off
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    # fmt: on

    n_stages = 6
    order = 5
    error_estimator_order = 4
