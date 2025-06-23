import sympy as sp
from sympy import Rational
import numpy as np


def compute_eigenvalues(matrix):
    """
    Computes the eigenvalues of a given sympy Matrix symbolically.

    Parameters:
        matrix (sp.Matrix): A square sympy Matrix

    Returns:
        list: A list of eigenvalues (symbolic expressions or numbers)
    """
    if not isinstance(matrix, sp.Matrix):
        raise TypeError("Input must be a sympy Matrix.")
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square.")

    # Compute eigenvalues
    eigenvals = matrix.eigenvals()
    return eigenvals


if __name__ == "__main__":
    # Example: 2x2 symbolic matrix
    # a, b, c, d = sp.symbols('a b c d')
    # A = sp.Matrix([[a, b],
    #                [c, d]])
    stages = 3
    match stages:
        case 1:
            A = sp.Matrix([[1.0]])
        case 2:
            # fmt: off
            A = sp.Matrix([[Rational(5, 12), Rational(-1, 12)],
                           [ Rational(3, 4), Rational(1, 4)]])
            # fmt: on
        case 3:
            sqrt6 = sp.sqrt(6)
            A = sp.Matrix(
                [
                    [
                        sp.Rational(88, 360) - sp.Rational(7, 360) * sqrt6,
                        sp.Rational(296, 1800) - sp.Rational(169, 1800) * sqrt6,
                        -sp.Rational(2, 225) + sp.Rational(3, 225) * sqrt6,
                    ],
                    [
                        sp.Rational(296, 1800) + sp.Rational(169, 1800) * sqrt6,
                        sp.Rational(88, 360) + sp.Rational(7, 360) * sqrt6,
                        -sp.Rational(2, 225) - sp.Rational(3, 225) * sqrt6,
                    ],
                    [
                        sp.Rational(16, 36) - sqrt6 / 36,
                        sp.Rational(16, 36) + sqrt6 / 36,
                        sp.Rational(1, 9),
                    ],
                ]
            )
        case _:
            raise NotImplementedError

    print("Matrix A:")
    sp.pprint(A)

    eigenvalues = compute_eigenvalues(A)
    eigenvectors = A.eigenvects()

    print("\nEigenvalues of A:")
    for val, mult in eigenvalues.items():
        val = sp.simplify(val)
        print(f"{val} (multiplicity {mult})")

    print("\nEigenvectors of A:")
    for vec, mult in eigenvectors.items():
        vec = sp.simplify(vec)
        print(f"{vec}")
