"""Stäckel potentials and separable potential implementations.

Stäckel potentials are separable potentials in confocal ellipsoidal coordinates
that allow for analytical solutions to the Jeans equations.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.special import ellipk, ellipe
from scipy.integrate import quad


class SeparablePotential:
    """Base class for separable potentials in triaxial systems."""
    
    def __init__(self, q1: float, q2: float, a: float = 1.0):
        """
        Initialize separable potential.
        
        Parameters
        ----------
        q1 : float
            Intermediate to major axis ratio (0 < q1 <= 1)
        q2 : float
            Minor to major axis ratio (0 < q2 <= q1)
        a : float, optional
            Semi-major axis length (default: 1.0)
        """
        if not (0 < q2 <= q1 <= 1):
            raise ValueError(f"Invalid axis ratios: q1={q1}, q2={q2}")
        
        self.q1 = q1
        self.q2 = q2
        self.a = a
        
        # Compute focal parameters
        self.a2 = (1.0 - q1**2) * a**2
        self.b2 = (1.0 - q2**2) * a**2
    
    def potential(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate gravitational potential at Cartesian coordinates."""
        raise NotImplementedError("Subclass must implement potential()")
    
    def acceleration(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute acceleration (force/mass) at Cartesian coordinates."""
        raise NotImplementedError("Subclass must implement acceleration()")
    
    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute density at Cartesian coordinates."""
        raise NotImplementedError("Subclass must implement density()")


class PowerLawStackelPotential(SeparablePotential):
    """
    Power-law Stäckel potential of the form Phi(lam, mu, nu) = sum of power-law terms.
    
    Based on de Zeeuw (1985), this potential has the form:
    Phi(lambda) = -G * M * lambda^(-alpha/2) / a^(alpha-1)
    
    and similarly for mu and nu coordinates.
    """
    
    def __init__(self, q1: float, q2: float, a: float = 1.0, alpha: float = 1.0):
        """
        Initialize power-law Stäckel potential.
        
        Parameters
        ----------
        q1 : float
            Intermediate to major axis ratio
        q2 : float
            Minor to major axis ratio
        a : float, optional
            Semi-major axis length (default: 1.0)
        alpha : float, optional
            Power-law index (default: 1.0 for 1/r potential)
        """
        super().__init__(q1, q2, a)
        self.alpha = alpha
        self.GM = 1.0  # Gravitational mass parameter (set to 1 for normalization)
    
    def _potential_ellipsoidal(
        self,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray
    ) -> np.ndarray:
        """Compute potential in ellipsoidal coordinates."""
        # Power-law potential components
        phi_lam = -self.GM * np.power(lam + 1.0, -self.alpha / 2)
        phi_mu = -self.GM * np.power(np.abs(mu) + self.a2, -self.alpha / 2)
        phi_nu = -self.GM * np.power(np.abs(nu) + self.b2, -self.alpha / 2)
        
        return phi_lam + phi_mu + phi_nu
    
    def potential(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate potential at Cartesian coordinates."""
        from .coordinates import cartesian_to_ellipsoidal
        
        lam, mu, nu = cartesian_to_ellipsoidal(x, y, z, self.q1, self.q2)
        return self._potential_ellipsoidal(lam, mu, nu)
    
    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute acceleration using numerical derivatives.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates
        
        Returns
        -------
        ax, ay, az : np.ndarray
            Acceleration components
        """
        h = 1e-6
        
        # Numerical gradients
        phi_xp = self.potential(x + h, y, z)
        phi_xm = self.potential(x - h, y, z)
        ax = -(phi_xp - phi_xm) / (2 * h)
        
        phi_yp = self.potential(x, y + h, z)
        phi_ym = self.potential(x, y - h, z)
        ay = -(phi_yp - phi_ym) / (2 * h)
        
        phi_zp = self.potential(x, y, z + h)
        phi_zm = self.potential(x, y, z - h)
        az = -(phi_zp - phi_zm) / (2 * h)
        
        return ax, ay, az
    
    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute density from potential using Poisson equation.
        Uses numerical Laplacian.
        """
        h = 1e-5
        
        # Compute Laplacian of potential numerically
        phi_0 = self.potential(x, y, z)
        
        phi_xp = self.potential(x + h, y, z)
        phi_xm = self.potential(x - h, y, z)
        lapl_x = (phi_xp - 2 * phi_0 + phi_xm) / h**2
        
        phi_yp = self.potential(x, y + h, z)
        phi_ym = self.potential(x, y - h, z)
        lapl_y = (phi_yp - 2 * phi_0 + phi_ym) / h**2
        
        phi_zp = self.potential(x, y, z + h)
        phi_zm = self.potential(x, y, z - h)
        lapl_z = (phi_zp - 2 * phi_0 + phi_zm) / h**2
        
        laplacian = lapl_x + lapl_y + lapl_z
        
        # Poisson equation: nabla^2(Phi) = 4*pi*G*rho
        # In units where 4*pi*G = 1:
        rho = laplacian / (4 * np.pi)
        
        return np.maximum(rho, 0)  # Ensure non-negative density


class HomogeneousTriaxialPotential(SeparablePotential):
    """
    Potential of a homogeneous triaxial ellipsoid.
    
    For a uniform density triaxial ellipsoid with semi-axes a > b > c,
    the interior potential is quadratic in the coordinates.
    """
    
    def __init__(self, q1: float, q2: float, a: float = 1.0, rho0: float = 1.0):
        """
        Initialize homogeneous triaxial potential.
        
        Parameters
        ----------
        q1 : float
            Intermediate to major axis ratio
        q2 : float
            Minor to major axis ratio
        a : float, optional
            Semi-major axis length
        rho0 : float, optional
            Central density
        """
        super().__init__(q1, q2, a)
        self.rho0 = rho0
    
    def potential(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Potential of homogeneous ellipsoid."""
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        # Normalized coordinates
        x_n = x / self.a
        y_n = y / (self.q1 * self.a)
        z_n = z / (self.q2 * self.a)
        
        # Compute elliptic integrals for the potential
        r_sq = x_n**2 + y_n**2 + z_n**2
        
        # Approximate potential (interior)
        # For a homogeneous ellipsoid, potential is more complex
        # This is a simplified version
        phi = -2 * np.pi * self.rho0 * (x_n**2 + y_n**2 + z_n**2)
        
        return phi
    
    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Acceleration for homogeneous ellipsoid."""
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        # Normalized coordinates
        x_n = x / self.a
        y_n = y / (self.q1 * self.a)
        z_n = z / (self.q2 * self.a)
        
        # For a uniform ellipsoid, acceleration is linear: a = -Omega^2 * r
        ax = -2 * np.pi * self.rho0 * x_n / self.a
        ay = -2 * np.pi * self.rho0 * y_n / (self.q1 * self.a)
        az = -2 * np.pi * self.rho0 * z_n / (self.q2 * self.a)
        
        return ax, ay, az
    
    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Density is uniform inside the ellipsoid."""
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        x_n = x / self.a
        y_n = y / (self.q1 * self.a)
        z_n = z / (self.q2 * self.a)
        
        # Inside ellipsoid: x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
        inside = x_n**2 + y_n**2 + z_n**2 <= 1.0
        
        return np.where(inside, self.rho0, 0.0)
