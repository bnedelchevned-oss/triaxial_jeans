"""
Differential operators in confocal ellipsoidal coordinates.

For orthogonal coordinates:

∇²Φ =
1/(h1 h2 h3) Σ_i ∂/∂q_i
[(h1 h2 h3 / h_i²) ∂Φ/∂q_i]

where:

q = (lambda, mu, nu)

and h_i are the Lamé coefficients.

Used by the Jeans equations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .geometry import EllipsoidalCoordinates


class EllipsoidalOperators:
    """
    Differential operators for the ellipsoidal coordinate system.
    """

    def __init__(self, coordinates: EllipsoidalCoordinates):
        self.coordinates = coordinates

    def derivative(self, f: Callable, q: tuple[float,float,float], i: int, eps: float = 1e-6) -> float:
        """
        Central finite difference derivative:

        ∂f/∂q_i
        """

        qp = list(q)
        qm = list(q)

        qp[i] += eps
        qm[i] -= eps

        return (f(*qp)-f(*qm))/(2*eps)

    def gradient(self, f: Callable, q: tuple[float,float,float], eps: float = 1e-6) -> np.ndarray:
        """
        Coordinate gradient components.
        """

        h = self.coordinates.scale_factors(*q)

        return np.array([
            self.derivative(f,q,i,eps)/h[i]
            for i in range(3)
        ])

    def laplacian(self, f: Callable, q: tuple[float,float,float], eps: float = 1e-6) -> float:
        """
        Laplacian in orthogonal ellipsoidal coordinates.
        """

        h = self.coordinates.scale_factors(*q)
        q = tuple(q)

        def term(i):
            def inner(*x):
                hi = self.coordinates.scale_factors(*x)
                return np.prod(hi)/hi[i]**2 * self.derivative(f,x,i,eps)

            return self.derivative(inner,q,i,eps)

        return sum(term(i) for i in range(3))/np.prod(h)
