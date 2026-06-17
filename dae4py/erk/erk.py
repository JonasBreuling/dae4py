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


def solve_ode(
    fun,
    t0,
    y0,
    t_span,
    t_eval=None,
    method=RK23,
    rtol=1e-3,
    atol=1e-6,
    h0=1e-3,
):
    # Do some sanity checks
    method.verify_tableau()

    A = method.A
    b = method.b
    b_hat = method.b_hat
    c = method.c
    n_stages = method.n_stages
    error_estimator_order = method.error_estimator_order
    error_exponent = -1 / (error_estimator_order + 1)

    # required for Hermite polynomial dense output
    Q = np.array([
        [0, 0, 1, 0],
        [-3, 3, -2, -1],
        [2, -2, 1, 1],
    ], dtype=float)

    n = len(y0)
    f0 = fun(t0, y0)
    Y_dot = np.empty((n_stages + 1, n), dtype=y0.dtype)

    n_steps = 0
    n_rejected = 0
    t = [t0]
    h = [h0]
    y = [y0.copy()]
    if t_eval is not None:
        t_eval_i = 0
        dense_output_exponent = np.arange(1, 4)[:, None]
        y_eval = []

    t1 = t0
    while t1 < t_span[-1]:
        n_steps += 1
        step_accepted = False
        step_rejected = False
        while not step_accepted:
            t1 = t0 + h0

            # stay in t_span
            if t1 + 1e-12 - t_span[-1] > 0:
                t1 = t_span[-1]

            h0 = t1 - t0

            # stage quadratures
            Y_dot[0] = f0  # FSAL
            for i in range(1, n_stages):
                Ti = t0 + c[i] * h0
                Yi = y0 + h0 * A[i, :i].dot(Y_dot[:i])
                Y_dot[i] = fun(Ti, Yi)

            # final quadrature
            y1 = y0 + h0 * b.dot(Y_dot[:-1])
            f1 = fun(t1, y1)
            Y_dot[-1] = f1

            # embedded method
            y1_hat = y0 + h0 * b_hat.dot(Y_dot)

            # error measure
            scale = atol + np.maximum(np.abs(y0), np.abs(y1)) * rtol
            error = y1_hat - y1
            error_norm = np.linalg.norm(error / scale) / error.size**0.5

            # step-size control
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm**error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h1 = h0 * factor

                step_accepted = True
            else:
                h0 *= max(MIN_FACTOR, SAFETY * error_norm**error_exponent)
                step_rejected = True
                n_rejected += 1

        t.append(t1)
        h.append(h0)
        y.append(y1.copy())

        # dense output
        if t_eval is not None:
            # rhs = np.vstack([y0, y1, h0 * f0, h0 * f1])
            # q = Q.dot(rhs)
            Z = np.vstack([y0, y1, h0 * f0, h0 * f1])
            q = Z.T @ Q.T

            # TODO: Document this
            t_eval_i1 = np.searchsorted(t_eval, t0 + h0, side="right")
            t_eval_step = t_eval[t_eval_i:t_eval_i1]

            if t_eval_step.size > 0:
                t_eval_i = t_eval_i1
                theta = (t_eval_step - t0) / h0
                theta_vec = theta**dense_output_exponent
                y_eval_step = y0[:, None] + q.dot(theta_vec)
                y_eval.append(y_eval_step.T)

        # bookkeeping
        h0 = h1
        t0 = t1
        f0 = f1.copy()
        y0 = y1.copy()

    return _RichResult(
        t=np.array(t),
        h=np.array(h),
        y=np.array(y),
        n_steps=n_steps,
        n_rejected=n_rejected,
        y_eval=np.concatenate(y_eval) if t_eval is not None else None,
    )
