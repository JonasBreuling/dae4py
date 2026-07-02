import numpy as np
from scipy._lib._util import _RichResult
from dae4py.erk.tableaus import RK23

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step-size.
MAX_FACTOR = 10  # Maximum allowed increase in a step-size.


def solve_ode(
    fun,
    y0,
    t_span,
    t_eval=None,
    method=RK23,
    rtol=1e-3,
    atol=1e-6,
    h0=1e-2,
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

    t0 = t_span[0]
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
            Z = np.vstack([y0, y1, h0 * f0, h0 * f1])
            q = Z.T @ Q.T

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
