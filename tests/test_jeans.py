import numpy as np

from triaxial_jeans.geometry import EllipsoidalCoordinates
from triaxial_jeans.stackel import StackelPotential
from triaxial_jeans.density import Density
from triaxial_jeans.jeans import JeansSolver


def test_jeans_point():

    coords = EllipsoidalCoordinates(-9,-4,-1)

    pot = StackelPotential(
        lambda t:-t*t
    )

    rho = Density(
        coords,
        pot
    )

    solver = JeansSolver(
        coords,
        pot,
        rho
    )

    result = solver.solve_point(
        10,
        2,
        .2
    )

    assert np.isfinite(result["density"])
    assert result["sigma2"].shape==(3,3)
