"""
Multi-Fidelity Gaussian Process Models

This module contains:
1. MFGP_Linear: Linear multi-fidelity GP (Emukit)
2. MFGP_NonLinear: Non-linear multi-fidelity GP (Emukit)
3. SingleFidelityGP: Standard GP for comparison

Extracted from your existing code in Documents 1 and 2.
"""

import numpy as np
import GPy
from typing import Tuple, Optional, Dict, Any

# Emukit imports
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel


# Benchmarking single-fidelity GP for comparison
class SingleFidelityGP:
    """
    Standard single-fidelity GP for baseline comparison.
    
    This is the LF GP from your Document 1 code.
    """
    
    def __init__(self, kernel_type: str = 'RBF', ARD: bool = True, 
                 num_restarts: int = 6):
        self.kernel_type = kernel_type
        self.ARD = ARD
        self.num_restarts = num_restarts
        self.model = None
        self.is_trained = False
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """
        Train GP on data.
        
        Args:
            X: Input locations (N, D)
            Y: Output values (N, 1)
        
        Returns:
            Training info dict
        """
        # Ensure correct shapes
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
        
        # Create kernel
        input_dim = X.shape[1]
        if self.kernel_type == 'RBF':
            kernel = GPy.kern.RBF(input_dim=input_dim, ARD=self.ARD)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Create and train model
        self.model = GPy.models.GPRegression(X, Y, kernel)
        
        # Constrain noise variance
        try:
            self.model.Gaussian_noise.variance.constrain_positive()
        except Exception:
            pass
        
        # Optimize with restarts
        self.model.optimize_restarts(num_restarts=self.num_restarts, verbose=True)
        
        self.is_trained = True
        
        return {
            'log_likelihood': float(self.model.log_likelihood()),
            'noise_variance': float(self.model.Gaussian_noise.variance),
            'lengthscales': self.model.kern.lengthscale.values.tolist(),
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at new locations.
        
        Args:
            X: Test locations (N, D)
            return_std: Whether to return standard deviation
        
        Returns:
            mean: Predicted mean (N, 1)
            std: Predicted std (N, 1) or None
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        mean, var = self.model.predict(X)
        
        if return_std:
            std = np.sqrt(var)
            return mean, std
        return mean, None


class MFGP_Linear:
    """
    Linear Multi-Fidelity Gaussian Process.
    
    Uses the linear autoregressive model:
        f_H(x) = rho * f_L(x) + delta(x)
    
    This is your code from Document 2.
    """
    
    def __init__(self, num_restarts: int = 10):
        self.num_restarts = num_restarts
        self.model = None
        self.gpy_model = None
        self.lf_gp = None  # Separate LF GP for predictions
        self.is_trained = False
        
        # Store scaling parameters
        self.rho = None
        self.bias = None
    
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray,
            compute_lf_scaling: bool = True) -> Dict[str, Any]:
        """
        Train linear MF-GP.
        
        Args:
            X_lf: LF input locations (N_L, D)
            Y_lf: LF output values (N_L, 1)
            X_hf: HF input locations (N_H, D)
            Y_hf: HF output values (N_H, 1)
            compute_lf_scaling: Whether to compute linear LF→HF scaling
        
        Returns:
            Training info dict
        """
        # Ensure correct shapes and types
        X_lf = np.asarray(X_lf, dtype=np.float64)
        Y_lf = np.asarray(Y_lf, dtype=np.float64).reshape(-1, 1)
        X_hf = np.asarray(X_hf, dtype=np.float64)
        Y_hf = np.asarray(Y_hf, dtype=np.float64).reshape(-1, 1)
        
        input_dim = X_lf.shape[1]
        
        # =====================================================
        # Step 1: Train LF GP (for later predictions)
        # =====================================================
        self.lf_gp = SingleFidelityGP(num_restarts=self.num_restarts)
        self.lf_gp.fit(X_lf, Y_lf)
        
        # =====================================================
        # Step 2: Compute LF→HF scaling
        # =====================================================
        if compute_lf_scaling:
            Y_lf_at_hf, _ = self.lf_gp.predict(X_hf, return_std=False)
            self.rho, self.bias = np.polyfit(Y_lf_at_hf.flatten(), Y_hf.flatten(), 1)
            # Apply calibration to ALL LF predictions
            Y_lf_pred_full, _ = self.lf_gp.predict(X_lf, return_std=False)
            Y_lf_calibrated = self.rho * Y_lf_pred_full + self.bias
        else:
            self.rho, self.bias = 1.0, 0.0
            Y_lf_calibrated = Y_lf

        # =====================================================
        # Step 3: Build MF training data
        # =====================================================
        # Combine LF and HF data with fidelity indicators.
        # Pass raw Y_lf (no pre-scaling): GPyLinearMultiFidelityModel
        # learns its own rho internally.
        Y_train = np.vstack([Y_lf_calibrated, Y_hf])
        X_train = np.vstack([X_lf, X_hf])
        
        # Fidelity column: 0 = LF, 1 = HF
        fidelities = np.vstack([
            np.zeros((X_lf.shape[0], 1)),
            np.ones((X_hf.shape[0], 1))
        ])
        X_train_with_fid = np.hstack([X_train, fidelities])
        
        # =====================================================
        # Step 4: Create and train MF-GP
        # =====================================================
        # Create kernels for each fidelity
        kernels = [GPy.kern.RBF(input_dim=input_dim, ARD=True) for _ in range(2)]
        lin_mf_kernel = LinearMultiFidelityKernel(kernels)
        
        # Create MF model
        self.gpy_model = GPyLinearMultiFidelityModel(
            X_train_with_fid, Y_train, lin_mf_kernel, n_fidelities=2
        )
        
        # Constrain noise
        try:
            self.gpy_model.mixed_noise.Gaussian_noise.constrain_positive()
            self.gpy_model.mixed_noise.Gaussian_noise_1.constrain_positive()
        except Exception:
            pass
        
        # Wrap and optimize
        self.model = GPyMultiOutputWrapper(
            self.gpy_model, 2, n_optimization_restarts=self.num_restarts
        )
        self.model.optimize()
        
        self.is_trained = True
        
        # Extract learned rho if available
        learned_rho = None
        try:
            learned_rho = self.gpy_model.kern.rho.values[0]
        except Exception:
            pass
        
        return {
            'scaling_rho': self.rho,
            'scaling_bias': self.bias,
            'learned_rho': learned_rho,
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict HF output at new locations.
        
        Args:
            X: Test locations (N, D)
            return_std: Whether to return standard deviation
        
        Returns:
            mean: Predicted HF mean (N, 1)
            std: Predicted HF std (N, 1) or None
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Add HF fidelity indicator (1)
        X_with_fid = np.hstack([X, np.ones((X.shape[0], 1))])
        
        mean, var = self.model.predict(X_with_fid)
        
        if return_std:
            std = np.sqrt(var)
            return mean, std
        return mean, None
    
    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output."""
        if self.lf_gp is None:
            raise RuntimeError("LF GP not available.")
        mean, _ = self.lf_gp.predict(X, return_std=False)
        return mean
    
    def predict_grid(self, x1_range: Tuple[float, float], 
                     x2_range: Tuple[float, float],
                     n_grid: int = 100) -> Dict[str, np.ndarray]:
        """
        Predict on a 2D grid for visualization.
        
        Returns:
            Dict with X1_grid, X2_grid, mean_grid, std_grid
        """
        x1_lin = np.linspace(x1_range[0], x1_range[1], n_grid)
        x2_lin = np.linspace(x2_range[0], x2_range[1], n_grid)
        X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
        X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        
        mean, std = self.predict(X_grid, return_std=True)
        
        return {
            'X1_grid': X1_grid,
            'X2_grid': X2_grid,
            'mean_grid': mean.reshape(n_grid, n_grid),
            'std_grid': std.reshape(n_grid, n_grid),
        }
