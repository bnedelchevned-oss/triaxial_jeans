"""
Confocal ellipsoidal coordinates.

Based on:
van de Ven et al. (2003)
General solution of the Jeans equations for triaxial galaxies
with separable potentials.

Coordinates (lambda, mu, nu) satisfy:

x²/(τ+α) + y²/(τ+β) + z²/(τ+γ) = 1

with:

α < β < γ < 0

The coordinate ranges are:

λ > -γ
-β < μ < -γ
-α < ν < -β
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


@dataclass(frozen=True)
class EllipsoidalCoordinates:
    """Confocal ellipsoidal coordinate system."""

    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        if not self.alpha < self.beta < self.gamma < 0:
            raise ValueError("Require alpha < beta < gamma < 0")

    def to_cartesian(self, lam: float, mu: float, nu: float,
                     signs: tuple[int,int,int]=(1,1,1)) -> tuple[float,float,float]:
        """
        Convert (lambda,mu,nu) to Cartesian coordinates.

        x²=((λ+α)(μ+α)(ν+α))/((α-β)(α-γ))
        and cyclic permutations.
        """

        a,b,c = self.alpha,self.beta,self.gamma

        x2 = (lam+a)*(mu+a)*(nu+a)/((a-b)*(a-c))
        y2 = (lam+b)*(mu+b)*(nu+b)/((b-a)*(b-c))
        z2 = (lam+c)*(mu+c)*(nu+c)/((c-a)*(c-b))

        if min(x2,y2,z2) < -1e-12:
            raise ValueError("Invalid coordinate ordering")

        return (
            signs[0]*np.sqrt(max(x2,0)),
            signs[1]*np.sqrt(max(y2,0)),
            signs[2]*np.sqrt(max(z2,0))
        )

    def _equation(self, tau: float, x: float, y: float, z: float) -> float:
        """Root equation defining lambda,mu,nu."""
        return (
            x*x/(tau+self.alpha)
            + y*y/(tau+self.beta)
            + z*z/(tau+self.gamma)
            - 1
        )

    def from_cartesian(self, x: float, y: float, z: float) -> tuple[float,float,float]:
        """
        Convert Cartesian position to ellipsoidal coordinates.
        """

        eps = 1e-12
        f = lambda t: self._equation(t,x,y,z)
        scale = max(1.0,x*x+y*y+z*z)

        lam = brentq(f,-self.gamma+eps,100*scale)
        mu = brentq(f,-self.beta+eps,-self.gamma-eps)
        nu = brentq(f,-self.alpha+eps,-self.beta-eps)

        return lam,mu,nu

    def scale_factors(self, lam: float, mu: float, nu: float) -> tuple[float,float,float]:
        """
        Lamé coefficients.

        h_lambda² =
        ((λ-μ)(λ-ν)) /
        (4(λ+α)(λ+β)(λ+γ))

        with cyclic permutations.
        """

        a,b,c = self.alpha,self.beta,self.gamma

        def h(t,u,v):
            return np.sqrt((t-u)*(t-v)/(4*(t+a)*(t+b)*(t+c)))

        return h(lam,mu,nu),h(mu,lam,nu),h(nu,lam,mu)

    def metric_tensor(self, lam: float, mu: float, nu: float) -> NDArray[np.float64]:
        """Return g_ij = diag(hλ²,hμ²,hν²)."""
        h1,h2,h3 = self.scale_factors(lam,mu,nu)
        return np.diag((h1*h1,h2*h2,h3*h3))

    def jacobian(self, lam: float, mu: float, nu: float) -> float:
        """Return volume element hλ hμ hν."""
        return np.prod(self.scale_factors(lam,mu,nu))
