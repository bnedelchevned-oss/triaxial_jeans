import numpy as np

from triaxial_jeans.quadrature import Quadrature


def test_integral():
    q = Quadrature()
    result = q.integrate(lambda x: x*x, 0, 1)
    assert np.isclose(result, 1/3, rtol=1e-10)


def test_gauss():
    q = Quadrature()
    result = q.gauss_legendre(lambda x: np.exp(x), 0, 1)
    assert np.isclose(result, np.e-1, rtol=1e-10)
