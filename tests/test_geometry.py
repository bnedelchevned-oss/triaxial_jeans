import numpy as np

from triaxial_jeans.geometry import EllipsoidalCoordinates


def test_round_trip():
    c = EllipsoidalCoordinates(-9,-4,-1)

    q = (10,2,0.2)

    xyz = c.to_cartesian(*q)
    q2 = c.from_cartesian(*xyz)

    assert np.allclose(q,q2,atol=1e-10)


def test_metric():
    c = EllipsoidalCoordinates(-9,-4,-1)

    g = c.metric_tensor(10,2,0.2)

    assert g.shape == (3,3)
    assert np.all(np.diag(g)>0)


def test_jacobian():
    c = EllipsoidalCoordinates(-9,-4,-1)

    assert c.jacobian(10,2,0.2)>0
