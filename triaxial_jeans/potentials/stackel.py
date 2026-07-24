"""Stäckel potentials and related separable models."""

import numpy as np
from typing import Tuple


class StackelPotential:
    """
    Stäckel potentials for triaxial systems.
    
    Stäckel potentials are separable potentials of the form:
    Phi(lambda, mu, nu) = Phi_lambda(lambda) + Phi_mu(mu) + Phi_nu(nu)
    
    where lambda, mu, nu are confocal ellipsoidal coordinates.
    """
    
    def __init__(self, a2: float, b2: float, c2: float):
        """
        Initialize a Stäckel potential.
        
        Parameters
        ----------
        a2 : float
            Semi-axis parameter (lambda coordinate)
        b2 : float
            Semi-axis parameter (mu coordinate)
        c2 : float
            Semi-axis parameter (nu coordinate)
        """
        self.a2 = a2
        self.b2 = b2
        self.c2 = c2
    
    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate the potential at Cartesian coordinates.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates
        
        Returns
        -------
        np.ndarray
            Potential values
        """
        raise NotImplementedError("Subclass must implement __call__")
    
    def acceleration(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute acceleration at Cartesian coordinates.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Cartesian coordinates
        
        Returns
        -------
        ax, ay, az : np.ndarray
            Acceleration components
        """
        raise NotImplementedError("Subclass must implement acceleration")
