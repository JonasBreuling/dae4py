import numpy as np
import pytest
from scipy.optimize import rosen_der, rosen_hess
from dae4py.math import newton, trust_region_newton

parameters_newton = ["2-point", "3-point", "cs", rosen_hess]


@pytest.mark.parametrize("jac,", parameters_newton)
def test_fsolve(jac):
    x0 = np.array([0.75, 1.85])

    sol = newton(rosen_der, x0, jac=jac, chord=False)
    x = sol.x
    print(sol)
    assert np.allclose(x, np.ones(2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("jac,", parameters_newton)
def test_trust_region_newton(jac):
    x0 = np.array([0.75, 1.85])

    sol = trust_region_newton(rosen_der, x0, jac=jac)
    x = sol.x
    print(sol)
    assert np.allclose(x, np.ones(2), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    for jac in parameters_newton:
        test_fsolve(jac=jac)
        test_trust_region_newton(jac=jac)
