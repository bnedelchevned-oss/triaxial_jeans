"""
Numerical integration utilities.

The Jeans solutions contain one-dimensional and multi-dimensional
integrals over ellipsoidal coordinates. This module provides wrappers
around scipy quadrature routines with consistent tolerances.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad, dblquad


class Quadrature:
    """Adaptive numerical quadrature settings."""

    def __init__(self, epsabs: float = 1e-10, epsrel: float = 1e-10, limit: int = 200):
        if epsabs <= 0 or epsrel <= 0:
            raise ValueError("Integration tolerances must be positive")
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> float:
        """Compute ∫ f(x) dx using adaptive Gauss-Kronrod quadrature."""
        value, error = quad(f, a, b, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)
        return value

    def integrate2d(self, f: Callable[[float, float], float], ax: float, bx: float,
                    ay: Callable[[float], float], by: Callable[[float], float]) -> float:
        """Compute a two-dimensional adaptive integral."""
        value, error = dblquad(f, ax, bx, ay, by, epsabs=self.epsabs, epsrel=self.epsrel)
        return value

    def gauss_legendre(self, f: Callable[[float], float], a: float, b: float, n: int = 64) -> float:
        """Fixed Gauss-Legendre quadrature."""
        x, w = np.polynomial.legendre.leggauss(n)
        t = 0.5*(b-a)*x + 0.5*(a+b)
        return 0.5*(b-a)*np.sum(w*f(t))
