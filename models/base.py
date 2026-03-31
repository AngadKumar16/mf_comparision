"""
Abstract Base Class for Multi-Fidelity Models

All models (GP, DNN, KAN, Hybrid) inherit from this class
to ensure a consistent API for training and prediction.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict, Any


class MFModelBase(ABC):
    """
    Base class for all multi-fidelity models.
    
    Provides unified interface:
        - fit(X_lf, Y_lf, X_hf, Y_hf) → training
        - predict(X, return_std=True) → HF predictions with uncertainty
        - predict_lf(X) → LF predictions
        - evaluate(X_test, Y_test) → metrics dict
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.training_history = []
    
    @abstractmethod
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model on LF and HF data.
        
        Args:
            X_lf: Low-fidelity inputs (N_L, D)
            Y_lf: Low-fidelity outputs (N_L, 1)
            X_hf: High-fidelity inputs (N_H, D)
            Y_hf: High-fidelity outputs (N_H, 1)
        
        Returns:
            Dict with training metrics/info
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, return_std: bool = True
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict HF output at new locations.
        
        Args:
            X: Test inputs (N, D)
            return_std: Whether to return uncertainty estimate
        
        Returns:
            mean: Predicted HF mean (N, 1)
            std: Predicted HF std (N, 1) or None
        """
        pass
    
    @abstractmethod
    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LF output at locations.
        
        Args:
            X: Test inputs (N, D)
        
        Returns:
            Y_lf: Predicted LF values (N, 1)
        """
        pass
    
    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test inputs
            Y_test: True test outputs
        
        Returns:
            Dict with RMSE, MAE, R², NLL, mean_std
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.name} not trained. Call fit() first.")
        
        Y_test = np.asarray(Y_test).reshape(-1, 1)
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        # Flatten for metrics
        y_true = Y_test.flatten()
        y_pred = y_pred.flatten()
        
        # Core metrics
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
        
        # Uncertainty metrics (if available)
        if y_std is not None and not np.all(y_std == 0):
            y_std = np.maximum(y_std.flatten(), 1e-6)
            nll = np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 
                         0.5 * ((y_true - y_pred) / y_std)**2)
            metrics['nll'] = nll
            metrics['mean_std'] = np.mean(y_std)
        
        return metrics
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.name}({status})"