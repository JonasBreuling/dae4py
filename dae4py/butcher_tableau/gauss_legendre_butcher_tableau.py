import numpy as np
from scipy._lib._util import _RichResult


def gauss_legendre_tableau(s):
    """
    Evaluate the Butcher tableau for the s-stage Gauss-Legendre method.

    Parameters
    ----------
    s: int
        Number of stages.

    Returns
    -------
    butcher_tableau: _RichResult
        Container that stores
            - A (array-like): Coefficient matrix.
            - b (array-like): Quadrature weights.
            - c (array-like): Quadrature nodes.
            - p (int): Classical order.
            - q (int): Stage order.
            - s (int): Number of stages.
    """
    # compute Gauss-Legendre quadrature nodes
    Poly = np.polynomial.Polynomial
    poly = Poly([0, 1]) ** s * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s)
    c = poly_der.roots()

    # compute weights from B(1), B(2), ..., B(s)
    V = np.vander(c, increasing=True)
    rhs = 1 / np.arange(1, s + 1)
    b = np.linalg.solve(V.T, rhs)

    # compute coefficent matrix
    R = np.diag(rhs)
    A = np.diag(c) @ V @ R @ np.linalg.inv(V)

    # quadrature and stage order
    p = 2 * s
    q = s

    return _RichResult(A=A, b=b, c=c, p=p, q=q, s=s)
