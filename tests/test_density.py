import numpy as np

from triaxial_jeans.geometry import EllipsoidalCoordinates
from triaxial_jeans.stackel import StackelPotential
from triaxial_jeans.density import Density


def test_density():

    coords = EllipsoidalCoordinates(
        -9,-4,-1
    )

    pot = StackelPotential(
        lambda t: -t*t
    )

    rho = Density(
        coords,
        pot
    )

    value = rho.density(
        10,
        2,
        0.2
    )

    assert np.isfinite(value)
