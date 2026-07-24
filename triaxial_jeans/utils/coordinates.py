"""Confocal ellipsoidal coordinates and coordinate transformations.

This module implements coordinate transformations between Cartesian and
confocal ellipsoidal coordinates, which are natural for triaxial systems.
"""

import numpy as np
from typing import Tuple, Union
from scipy.optimize import fsolve, brentq


def get_confocal_parameters(q1: float, q2: float) -> Tuple[float, float]:
    """
    Compute confocal focal parameters from axis ratios.
    
    For a triaxial ellipsoid with semi-axes a > b > c, where:
    - q1 = b/a (intermediate to major axis ratio)
    - q2 = c/a (minor to major axis ratio)
    
    The focal parameters are:
    - a2 = a^2 * (1 - q1^2)
    - b2 = a^2 * (1 - q2^2)
    
    Parameters
    ----------
    q1 : float
        Intermediate to major axis ratio (0 < q1 <= 1)
    q2 : float
        Minor to major axis ratio (0 < q2 <= q1)
    
    Returns
    -------
    a2, b2 : float
        Focal parameters (normalized to a^2 = 1)
    """
    if not (0 < q2 <= q1 <= 1):
        raise ValueError(f"Invalid axis ratios: q1={q1}, q2={q2}. Require 0 < q2 <= q1 <= 1")
    
    a2 = 1.0 - q1**2  # Normalized by a^2
    b2 = 1.0 - q2**2  # Normalized by a^2
    
    return a2, b2


def cartesian_to_ellipsoidal(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    q1: float,
    q2: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to confocal ellipsoidal coordinates.
    
    The confocal ellipsoidal coordinates (lambda, mu, nu) are defined such that:
    - Surfaces of constant lambda are confocal ellipsoids
    - Surfaces of constant mu are confocal hyperboloids of one sheet
    - Surfaces of constant nu are confocal hyperboloids of two sheets
    
    Relationship to Cartesian coordinates:
    x^2/(lambda + a2) + y^2/(lambda + b2) + z^2/lambda = 1
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates (normalized by a = 1)
    q1 : float
        Intermediate to major axis ratio (0 < q1 <= 1)
    q2 : float
        Minor to major axis ratio (0 < q2 <= q1)
    
    Returns
    -------
    lam, mu, nu : np.ndarray
        Confocal ellipsoidal coordinates
    """
    a2, b2 = get_confocal_parameters(q1, q2)
    
    # Ensure inputs are numpy arrays
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    
    # For a single point, solve the system
    def ellipsoidal_coords(point):
        xi, yi, zi = point
        
        def eq_lambda(lam):
            return xi**2 / (lam + a2) + yi**2 / (lam + b2) + zi**2 / lam - 1.0
        
        def eq_mu(mu):
            return xi**2 / (mu + a2) + yi**2 / (mu + b2) + zi**2 / mu - 1.0
        
        def eq_nu(nu):
            return xi**2 / (nu + a2) + yi**2 / (nu + b2) + zi**2 / nu - 1.0
        
        # Find roots: lambda > 0, -a2 < mu < 0, -b2 < nu < -a2
        try:
            lam = brentq(eq_lambda, 0, 1e6)
        except ValueError:
            lam = fsolve(eq_lambda, 0.1)[0]
        
        try:
            mu = brentq(eq_mu, -a2 + 1e-8, -1e-8)
        except ValueError:
            mu = fsolve(eq_mu, -a2/2)[0]
        
        try:
            nu = brentq(eq_nu, -b2 + 1e-8, -a2 - 1e-8)
        except ValueError:
            nu = fsolve(eq_nu, -b2/2)[0]
        
        return lam, mu, nu
    
    # Vectorize for multiple points
    lam = np.zeros_like(x)
    mu = np.zeros_like(x)
    nu = np.zeros_like(x)
    
    flat_x = x.flatten()
    flat_y = y.flatten()
    flat_z = z.flatten()
    
    for i in range(len(flat_x)):
        lam_i, mu_i, nu_i = ellipsoidal_coords((flat_x[i], flat_y[i], flat_z[i]))
        lam.flat[i] = lam_i
        mu.flat[i] = mu_i
        nu.flat[i] = nu_i
    
    return lam, mu, nu


def ellipsoidal_to_cartesian(
    lam: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    q1: float,
    q2: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert confocal ellipsoidal coordinates to Cartesian coordinates.
    
    Parameters
    ----------
    lam, mu, nu : np.ndarray
        Confocal ellipsoidal coordinates
    q1 : float
        Intermediate to major axis ratio (0 < q1 <= 1)
    q2 : float
        Minor to major axis ratio (0 < q2 <= q1)
    
    Returns
    -------
    x, y, z : np.ndarray
        Cartesian coordinates
    """
    a2, b2 = get_confocal_parameters(q1, q2)
    
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    
    # Direct inversion formulas
    denominator = (lam + a2) * (mu + a2) * (nu + a2)
    
    x_sq = (lam * mu * nu) / ((a2 - b2) * a2)
    y_sq = (lam * mu * nu) / ((a2 - b2) * b2)
    z_sq = -(lam * mu * nu) / ((a2 - b2) * 0.0) if a2 == b2 else (lam * mu * nu) / (a2 * b2)
    
    # Corrected formulation
    x = np.sqrt(np.maximum(lam * (mu + a2) * (nu + a2) / ((lam + a2) * (mu - nu) * (nu - lam)), 0))
    y = np.sqrt(np.maximum(mu * (lam + b2) * (nu + b2) / ((mu + b2) * (nu - mu) * (lam - mu)), 0))
    z = np.sqrt(np.maximum(nu * (lam + b2) * (mu + b2) / ((nu + b2) * (mu - nu) * (lam - nu)), 0))
    
    return x, y, z


def scale_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale coordinates by semi-major axis length.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Normalized coordinates (a = 1)
    a : float
        Semi-major axis length
    
    Returns
    -------
    x_scaled, y_scaled, z_scaled : np.ndarray
        Scaled coordinates
    """
    return a * x, a * y, a * z


def jacobian_ellipsoidal(
    lam: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    q1: float,
    q2: float
) -> np.ndarray:
    """
    Compute Jacobian determinant for coordinate transformation.
    
    Parameters
    ----------
    lam, mu, nu : np.ndarray
        Confocal ellipsoidal coordinates
    q1 : float
        Intermediate to major axis ratio
    q2 : float
        Minor to major axis ratio
    
    Returns
    -------
    jacobian : np.ndarray
        Jacobian determinant |d(x,y,z)/d(lam,mu,nu)|
    """
    a2, b2 = get_confocal_parameters(q1, q2)
    
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    
    # Jacobian for ellipsoidal coordinates
    numerator = lam * mu * nu
    denominator = 8 * (lam + a2) * (mu + b2) * (nu) * \
                  np.sqrt((lam + a2) * (mu + a2) * (nu + a2)) * \
                  np.sqrt((lam + b2) * (mu + b2) * (nu + b2)) * \
                  np.sqrt(lam * mu * nu)
    
    jacobian = np.abs(numerator / denominator)
    
    return jacobian
