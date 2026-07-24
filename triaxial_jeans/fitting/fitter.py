"""Model fitting utilities for triaxial Jeans models."""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from scipy.optimize import minimize, least_squares
from scipy.stats import chi2


class JeansFitter:
    """
    Fit triaxial Jeans models to observational kinematics data.
    """
    
    def __init__(
        self,
        potential,
        jeans_solver,
        observations: Dict[str, np.ndarray]
    ):
        """
        Initialize model fitter.
        
        Parameters
        ----------
        potential : SeparablePotential
            Gravitational potential
        jeans_solver : JeansSolver
            Jeans equation solver
        observations : dict
            Observational data with keys:
            - 'position': (N, 3) array of positions
            - 'velocity': (N, 3) array of velocities
            - 'dispersion': (N,) array of velocity dispersions
            - 'error': (N,) array of measurement errors
        """
        self.potential = potential
        self.jeans_solver = jeans_solver
        self.observations = observations
    
    def model_predictions(
        self,
        params: np.ndarray
    ) -> np.ndarray:
        """
        Compute model predictions at observation locations.
        
        Parameters
        ----------
        params : np.ndarray
            Model parameters (e.g., mass normalization, axis ratios)
        
        Returns
        -------
        np.ndarray
            Predicted velocity dispersions
        """
        positions = self.observations['position']
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Apply parameter scaling
        results = self.jeans_solver.solve_jeans_equations(x, y, z)
        
        # Compute line-of-sight dispersion or other observable
        sigma_pred = np.sqrt(results['sigma_x']**2 + 
                            results['sigma_y']**2 + 
                            results['sigma_z']**2) / np.sqrt(3)
        
        return sigma_pred
    
    def residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute residuals between model and observations.
        
        Parameters
        ----------
        params : np.ndarray
            Model parameters
        
        Returns
        -------
        np.ndarray
            Residuals (data - model) / error
        """
        predictions = self.model_predictions(params)
        observations = self.observations['dispersion']
        errors = self.observations['error']
        
        residuals = (observations - predictions) / (errors + 1e-20)
        
        return residuals
    
    def chi_squared(self, params: np.ndarray) -> float:
        """
        Compute chi-squared statistic.
        
        Parameters
        ----------
        params : np.ndarray
            Model parameters
        
        Returns
        -------
        float
            Chi-squared value
        """
        res = self.residuals(params)
        return np.sum(res**2)
    
    def fit(
        self,
        initial_params: np.ndarray,
        method: str = 'least_squares',
        bounds: Optional[Tuple] = None,
        **kwargs
    ) -> Dict:
        """
        Fit model to observations.
        
        Parameters
        ----------
        initial_params : np.ndarray
            Initial parameter values
        method : str, optional
            Fitting method ('least_squares', 'minimize', 'lm')
        bounds : tuple, optional
            Parameter bounds ((lower,), (upper,))
        **kwargs
            Additional arguments passed to optimizer
        
        Returns
        -------
        dict
            Fitting results including:
            - 'params': optimal parameters
            - 'chi2': chi-squared value
            - 'success': whether fit converged
            - 'message': fit message
        """
        if method == 'least_squares':
            result = least_squares(
                self.residuals,
                initial_params,
                bounds=bounds,
                **kwargs
            )
            return {
                'params': result.x,
                'chi2': np.sum(result.fun**2),
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nfev,
            }
        
        elif method == 'minimize':
            result = minimize(
                self.chi_squared,
                initial_params,
                bounds=bounds,
                method='L-BFGS-B',
                **kwargs
            )
            return {
                'params': result.x,
                'chi2': result.fun,
                'success': result.success,
                'message': result.message,
            }
        
        else:
            raise ValueError(f"Unknown fitting method: {method}")
    
    def parameter_covariance(
        self,
        params: np.ndarray,
        method: str = 'numerical'
    ) -> np.ndarray:
        """
        Estimate parameter covariance matrix.
        
        Parameters
        ----------
        params : np.ndarray
            Best-fit parameters
        method : str, optional
            Method for computing covariance ('numerical' or 'jacobian')
        
        Returns
        -------
        np.ndarray
            Parameter covariance matrix
        """
        if method == 'numerical':
            h = 1e-6
            n_params = len(params)
            
            jacobian = np.zeros((len(self.observations['dispersion']), n_params))
            
            for i in range(n_params):
                params_plus = params.copy()
                params_plus[i] += h
                
                res_plus = self.residuals(params_plus)
                res = self.residuals(params)
                
                jacobian[:, i] = (res_plus - res) / h
            
            # Covariance = (J^T J)^-1
            jt_j = np.dot(jacobian.T, jacobian)
            covariance = np.linalg.inv(jt_j)
            
            return covariance
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def parameter_errors(self, params: np.ndarray) -> np.ndarray:
        """
        Estimate parameter errors (1-sigma).
        
        Parameters
        ----------
        params : np.ndarray
            Best-fit parameters
        
        Returns
        -------
        np.ndarray
            Parameter errors
        """
        covariance = self.parameter_covariance(params)
        errors = np.sqrt(np.diagonal(covariance))
        return errors


class BootstrapFitter:
    """
    Perform bootstrap resampling to estimate parameter uncertainties.
    """
    
    def __init__(self, fitter: JeansFitter, n_bootstrap: int = 100):
        """
        Initialize bootstrap fitter.
        
        Parameters
        ----------
        fitter : JeansFitter
            Original fitter object
        n_bootstrap : int, optional
            Number of bootstrap samples
        """
        self.fitter = fitter
        self.n_bootstrap = n_bootstrap
    
    def run(self, initial_params: np.ndarray) -> Dict:
        """
        Run bootstrap analysis.
        
        Parameters
        ----------
        initial_params : np.ndarray
            Initial parameters for fitting
        
        Returns
        -------
        dict
            Bootstrap results with:
            - 'params_samples': parameter samples
            - 'param_mean': mean parameters
            - 'param_std': standard deviations
            - 'param_percentiles': percentiles
        """
        n_data = len(self.fitter.observations['dispersion'])
        params_samples = []
        
        for i in range(self.n_bootstrap):
            # Resample indices
            indices = np.random.choice(n_data, size=n_data, replace=True)
            
            # Create bootstrap sample
            bootstrap_obs = {}
            for key, value in self.fitter.observations.items():
                bootstrap_obs[key] = value[indices]
            
            # Fit bootstrap sample
            old_obs = self.fitter.observations
            self.fitter.observations = bootstrap_obs
            
            result = self.fitter.fit(initial_params)
            params_samples.append(result['params'])
            
            self.fitter.observations = old_obs
        
        params_samples = np.array(params_samples)
        
        return {
            'params_samples': params_samples,
            'param_mean': np.mean(params_samples, axis=0),
            'param_std': np.std(params_samples, axis=0),
            'param_percentiles': {
                '16': np.percentile(params_samples, 16, axis=0),
                '50': np.percentile(params_samples, 50, axis=0),
                '84': np.percentile(params_samples, 84, axis=0),
            },
        }
