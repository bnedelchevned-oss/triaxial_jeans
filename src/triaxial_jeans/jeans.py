import numpy as np

from triaxial_jeans.geometry import EllipsoidalCoordinates
from triaxial_jeans.operators import EllipsoidalOperators


def test_gradient():

    c = EllipsoidalCoordinates(-9,-4,-1)
    op = EllipsoidalOperators(c)

    grad = op.gradient(
        lambda l,m,n: l*l+m*m+n*n,
        (10,2,.2)
    )

    assert grad.shape == (3,)
    assert np.all(np.isfinite(grad))


def test_laplacian():

    c = EllipsoidalCoordinates(-9,-4,-1)
    op = EllipsoidalOperators(c)

    result = op.laplacian(
        lambda l,m,n: l*l+m*m+n*n,
        (10,2,.2)
    )

    assert np.isfinite(result)
