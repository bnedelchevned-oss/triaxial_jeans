"""
Triaxial Jeans: Solutions for Triaxial Galaxies with Separable Potentials

A Python package for solving the Jeans equations in triaxial potentials,
following de Zeeuw, Bureau, & Franx (2002).

Reference:
    de Zeeuw, T., Bureau, M., & Franx, M. (2002).
    "General solution of the Jeans equations for triaxial galaxies
    with separable potentials."
    The Astrophysical Journal, 343(1), 3-21.
"""

__version__ = "0.1.0"
__author__ = "Nedělchev Nikolay"

# Import main modules
from . import potentials  # noqa: F401
from . import jeans  # noqa: F401
from . import kinematics  # noqa: F401
from . import fitting  # noqa: F401
from . import utils  # noqa: F401

__all__ = [
    "potentials",
    "jeans",
    "kinematics",
    "fitting",
    "utils",
]
