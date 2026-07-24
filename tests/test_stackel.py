import numpy as np

from triaxial_jeans.stackel import StackelPotential


def test_constant_generator():

    p = StackelPotential(lambda x: x*x)

    phi = p.potential(10,2,0.2)

    assert np.isfinite(phi)


def test_derivatives():

    p = StackelPotential(lambda x: x*x)

    d = p.derivatives(10,2,0.2)

    assert len(d)==3
    assert np.all(np.isfinite(d))


def test_force():

    p = StackelPotential(lambda x: x*x)

    f = p.force(10,2,0.2)

    assert len(f)==3
