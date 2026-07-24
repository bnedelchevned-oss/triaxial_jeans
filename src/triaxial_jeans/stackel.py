"""
Stäckel potentials.

Based on:

de Zeeuw (1985)
and van de Ven et al. (2003)

The separable triaxial potential is:

Φ =
-f(λ)/((λ-μ)(λ-ν))
-f(μ)/((μ-ν)(μ-λ))
-f(ν)/((ν-λ)(ν-μ))

where f(τ) is the Stäckel generating function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class StackelPotential:
    """
    Triaxial Stäckel potential.

    Parameters
    ----------
    f:
        Generating function f(τ).

    G:
        Gravitational constant.
    """

    f: Callable[[float], float]
    G: float = 1.0

    def potential(self, lam: float, mu: float, nu: float) -> float:
        """
        Evaluate the separable Stäckel potential.
        """

        f = self.f

        term_l = -f(lam)/((lam-mu)*(lam-nu))
        term_m = -f(mu)/((mu-nu)*(mu-lam))
        term_n = -f(nu)/((nu-lam)*(nu-mu))

        return term_l + term_m + term_n

    def derivatives(self, lam: float, mu: float, nu: float,
                    eps: float = 1e-6) -> tuple[float,float,float]:
        """
        Numerical derivatives:

        ∂Φ/∂λ, ∂Φ/∂μ, ∂Φ/∂ν

        Used initially for validation.
        """

        x = np.array([lam,mu,nu],dtype=float)

        grad = []

        for i in range(3):
            xp = x.copy()
            xm = x.copy()

            xp[i] += eps
            xm[i] -= eps

            grad.append(
                (
                    self.potential(*xp)
                    -
                    self.potential(*xm)
                )/(2*eps)
            )

        return tuple(grad)

    def force(self, lam: float, mu: float, nu: float) -> tuple[float,float,float]:
        """
        Coordinate force:

        F_i = -∂Φ/∂q_i
        """

        dphi = self.derivatives(lam,mu,nu)

        return tuple(-x for x in dphi)
