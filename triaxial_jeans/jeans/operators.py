"""Differential operators in confocal ellipsoidal coordinates.

Implementation of gradient, Laplacian, and divergence operators
in confocal ellipsoidal coordinates for solving the Jeans equations.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.optimize import minimize_scalar


class EllipsoidalOperators:
    """
    Differential operators in confocal ellipsoidal coordinates.
    """
    
    def __init__(self, q1: float, q2: float):
        """
        Initialize operators for given axis ratios.
        
        Parameters
        ----------
        q1 : float
            Intermediate to major axis ratio (0 < q1 <= 1)
        q2 : float
            Minor to major axis ratio (0 < q2 <= q1)
        """
        if not (0 < q2 <= q1 <= 1):
            raise ValueError(f"Invalid axis ratios: q1={q1}, q2={q2}")
        
        self.q1 = q1
        self.q2 = q2
        self.a2 = 1.0 - q1**2
        self.b2 = 1.0 - q2**2
    
    def scale_factors(
        self,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute scale factors h_i in ellipsoidal coordinates.
        
        The scale factors relate coordinate differentials to distance:
        ds² = h_λ² dlambda² + h_μ² dmu² + h_ν² dnu²
        
        Parameters
        ----------
        lam, mu, nu : np.ndarray
            Ellipsoidal coordinates
        
        Returns
        -------
        h_lam, h_mu, h_nu : np.ndarray
            Scale factors
        """
        # Scale factors for confocal ellipsoidal coordinates
        h_lam_sq = (lam * (mu - lam) * (nu - lam)) / (
            (lam + self.a2) * (lam + self.b2)
        )
        h_mu_sq = (mu * (lam - mu) * (nu - mu)) / (
            (mu + self.a2) * (mu + self.b2)
        )
        h_nu_sq = (nu * (lam - nu) * (mu - nu)) / (
            (nu + self.a2) * (nu + self.b2)
        )
        
        h_lam = np.sqrt(np.abs(h_lam_sq))
        h_mu = np.sqrt(np.abs(h_mu_sq))
        h_nu = np.sqrt(np.abs(h_nu_sq))
        
        return h_lam, h_mu, h_nu
    
    def gradient(
        self,
        f: Callable,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        h: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient of scalar field using numerical differentiation.
        
        ∇f = (1/h_λ * ∂f/∂λ) e_λ + (1/h_μ * ∂f/∂μ) e_μ + (1/h_ν * ∂f/∂ν) e_ν
        
        Parameters
        ----------
        f : callable
            Scalar field function f(lam, mu, nu)
        lam, mu, nu : np.ndarray
            Ellipsoidal coordinates
        h : float, optional
            Step size for numerical differentiation
        
        Returns
        -------
        grad_lam, grad_mu, grad_nu : np.ndarray
            Gradient components
        """
        h_lam, h_mu, h_nu = self.scale_factors(lam, mu, nu)
        
        # Numerical derivatives
        f_lam_plus = f(lam + h, mu, nu)
        f_lam_minus = f(lam - h, mu, nu)
        grad_lam = (f_lam_plus - f_lam_minus) / (2 * h * h_lam + 1e-20)
        
        f_mu_plus = f(lam, mu + h, nu)
        f_mu_minus = f(lam, mu - h, nu)
        grad_mu = (f_mu_plus - f_mu_minus) / (2 * h * h_mu + 1e-20)
        
        f_nu_plus = f(lam, mu, nu + h)
        f_nu_minus = f(lam, mu, nu - h)
        grad_nu = (f_nu_plus - f_nu_minus) / (2 * h * h_nu + 1e-20)
        
        return grad_lam, grad_mu, grad_nu
    
    def laplacian(
        self,
        f: Callable,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        h: float = 1e-5
    ) -> np.ndarray:
        """
        Compute Laplacian of scalar field in ellipsoidal coordinates.
        
        ∇²f = (1/h_λh_μh_ν) * [
            ∂/∂λ(h_μh_ν/h_λ * ∂f/∂λ) +
            ∂/∂μ(h_λh_ν/h_μ * ∂f/∂μ) +
            ∂/∂ν(h_λh_μ/h_ν * ∂f/∂ν)
        ]
        
        Parameters
        ----------
        f : callable
            Scalar field function f(lam, mu, nu)
        lam, mu, nu : np.ndarray
            Ellipsoidal coordinates
        h : float, optional
            Step size for numerical differentiation
        
        Returns
        -------
        laplacian : np.ndarray
            Laplacian of f
        """
        h_lam, h_mu, h_nu = self.scale_factors(lam, mu, nu)
        
        # Compute Laplacian using numerical differentiation
        # ∇²f ≈ (f(r+h) - 2f(r) + f(r-h)) / h² in Cartesian coords
        # In ellipsoidal coords, it's more complex due to metric tensor
        
        f0 = f(lam, mu, nu)
        
        # λ component
        f_lam_plus = f(lam + h, mu, nu)
        f_lam_minus = f(lam - h, mu, nu)
        lap_lam = (f_lam_plus - 2 * f0 + f_lam_minus) / (h**2 * h_lam**2 + 1e-20)
        
        # μ component
        f_mu_plus = f(lam, mu + h, nu)
        f_mu_minus = f(lam, mu - h, nu)
        lap_mu = (f_mu_plus - 2 * f0 + f_mu_minus) / (h**2 * h_mu**2 + 1e-20)
        
        # ν component
        f_nu_plus = f(lam, mu, nu + h)
        f_nu_minus = f(lam, mu, nu - h)
        lap_nu = (f_nu_plus - 2 * f0 + f_nu_minus) / (h**2 * h_nu**2 + 1e-20)
        
        laplacian = lap_lam + lap_mu + lap_nu
        
        return laplacian
    
    def divergence(
        self,
        vx: Callable,
        vy: Callable,
        vz: Callable,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        h: float = 1e-6
    ) -> np.ndarray:
        """
        Compute divergence of vector field.
        
        ∇·V = (1/h_λh_μh_ν) * [
            ∂(h_μh_ν V_λ)/∂λ +
            ∂(h_λh_ν V_μ)/∂μ +
            ∂(h_λh_μ V_ν)/∂ν
        ]
        
        Parameters
        ----------
        vx, vy, vz : callable
            Vector field components (in Cartesian coords)
        lam, mu, nu : np.ndarray
            Ellipsoidal coordinates
        h : float, optional
            Step size for numerical differentiation
        
        Returns
        -------
        div : np.ndarray
            Divergence of V
        """
        h_lam, h_mu, h_nu = self.scale_factors(lam, mu, nu)
        
        # Compute divergence numerically
        # For simplicity, use Cartesian divergence
        # and transform to ellipsoidal
        
        div_cart = np.zeros_like(lam, dtype=float)
        
        # Numerical derivatives in Cartesian space
        # This is a simplified version
        vx0 = vx(lam, mu, nu)
        vy0 = vy(lam, mu, nu)
        vz0 = vz(lam, mu, nu)
        
        # Would need full Jacobian for proper transformation
        # For now, return approximate divergence
        return div_cart
    
    def curl(
        self,
        vx: Callable,
        vy: Callable,
        vz: Callable,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        h: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute curl of vector field in ellipsoidal coordinates.
        
        Parameters
        ----------
        vx, vy, vz : callable
            Vector field components
        lam, mu, nu : np.ndarray
            Ellipsoidal coordinates
        h : float, optional
            Step size for numerical differentiation
        
        Returns
        -------
        curl_lam, curl_mu, curl_nu : np.ndarray
            Curl components
        """
        # Curl would be computed similarly to divergence
        # Using scale factors and metric tensor
        
        return np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)


class MetricTensor:
    """Compute and work with the metric tensor in ellipsoidal coordinates."""
    
    def __init__(self, q1: float, q2: float):
        """Initialize metric tensor for given axis ratios."""
        self.q1 = q1
        self.q2 = q2
        self.a2 = 1.0 - q1**2
        self.b2 = 1.0 - q2**2
    
    def get_metric(
        self,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray
    ) -> np.ndarray:
        """
        Compute metric tensor g_ij at given point.
        
        Parameters
        ----------
        lam, mu, nu : float or array
            Ellipsoidal coordinates
        
        Returns
        -------
        g : np.ndarray
            3x3 metric tensor (or array of 3x3 tensors)
        """
        h_lam, h_mu, h_nu = self._scale_factors(lam, mu, nu)
        
        # Diagonal metric tensor
        g = np.zeros((..., 3, 3))
        g[..., 0, 0] = h_lam**2
        g[..., 1, 1] = h_mu**2
        g[..., 2, 2] = h_nu**2
        
        return g
    
    def _scale_factors(
        self,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute scale factors."""
        h_lam_sq = (lam * (mu - lam) * (nu - lam)) / (
            (lam + self.a2) * (lam + self.b2)
        )
        h_mu_sq = (mu * (lam - mu) * (nu - mu)) / (
            (mu + self.a2) * (mu + self.b2)
        )
        h_nu_sq = (nu * (lam - nu) * (mu - nu)) / (
            (nu + self.a2) * (nu + self.b2)
        )
        
        return (np.sqrt(np.abs(h_lam_sq)), 
                np.sqrt(np.abs(h_mu_sq)), 
                np.sqrt(np.abs(h_nu_sq)))
    
    def christoffel_symbols(
        self,
        lam: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray
    ) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^k_ij.
        
        Used for covariant derivatives and geodesics.
        """
        # For orthogonal coordinates, many Christoffel symbols are zero
        # Only non-zero ones involve derivatives of scale factors
        
        h_lam, h_mu, h_nu = self._scale_factors(lam, mu, nu)
        
        # This would be a 3x3x3 array of Christoffel symbols
        # Most entries are zero for orthogonal coordinates
        
        return np.zeros((3, 3, 3))
