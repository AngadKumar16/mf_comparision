"""
Deep Ensemble for Uncertainty Quantification

Trains M independent models with different random seeds
and aggregates predictions for uncertainty estimation.

Reference: Lakshminarayanan et al., "Simple and Scalable 
Predictive Uncertainty Estimation using Deep Ensembles" (2017)
"""

import numpy as np
import tensorflow as tf
from typing import Callable, List, Tuple, Optional, Dict, Any


class DeepEnsemble:
    """
    Deep Ensemble wrapper for any model.
    
    Usage:
        factory = lambda: MFDNN(layers_lf=[2,20,1], ...)
        ensemble = DeepEnsemble(factory, n_models=5)
        ensemble.fit(X_lf, Y_lf, X_hf, Y_hf)
        mean, epistemic, aleatoric = ensemble.predict_with_decomposition(X_test)
    """
    
    def __init__(self, model_factory: Callable, n_models: int = 5):
        """
        Args:
            model_factory: Function that returns a new model instance
            n_models: Number of ensemble members (5-10 typical)
        """
        self.model_factory = model_factory
        self.n_models = n_models
        self.models: List = []
        self.is_trained = False
        self.name = f"Ensemble({n_models})"
    
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray,
            verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train ensemble with different random seeds.
        
        Each member is initialized with a different seed for:
        - Different weight initialization
        - Different data shuffling (if applicable)
        """
        self.models = []
        training_infos = []
        
        for i in range(self.n_models):
            if verbose:
                print(f"\n--- Training ensemble member {i+1}/{self.n_models} ---")
            
            # Set different seeds
            seed = 42 + i * 1000
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
            
            # Create and train model
            model = self.model_factory()
            
            # Reduce verbosity for individual models
            if hasattr(model, 'verbose'):
                model.verbose = False
            
            info = model.fit(X_lf, Y_lf, X_hf, Y_hf, **kwargs)
            
            self.models.append(model)
            training_infos.append(info)
            
            if verbose and 'final_loss' in info:
                print(f"  Member {i+1} final loss: {info['final_loss']:.6f}")
        
        self.is_trained = True
        
        return {
            'n_models': self.n_models,
            'member_infos': training_infos,
            'avg_final_loss': np.mean([
                info.get('final_loss', 0) for info in training_infos
            ])
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Ensemble prediction with epistemic uncertainty.
        
        Returns:
            mean: Average of ensemble predictions
            std: Std across ensemble members (epistemic uncertainty)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        # Collect predictions from all members
        predictions = []
        for model in self.models:
            y_pred, _ = model.predict(X, return_std=False)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)  # (n_models, N, 1)
        
        mean = np.mean(predictions, axis=0)
        
        if return_std:
            std = np.std(predictions, axis=0)
            return mean, std
        return mean, None
    
    def predict_with_decomposition(self, X: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full uncertainty decomposition.
        
        Returns:
            mean: Ensemble mean prediction
            epistemic: Uncertainty from model disagreement (variance of means)
            aleatoric: Average predicted noise (mean of variances, if available)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        all_means = []
        all_vars = []
        
        for model in self.models:
            mu, sigma = model.predict(X, return_std=True)
            all_means.append(mu)
            if sigma is not None:
                all_vars.append(sigma**2)
        
        all_means = np.array(all_means)  # (n_models, N, 1)
        
        # Ensemble statistics
        mean = np.mean(all_means, axis=0)
        
        # Epistemic: variance of the means (model uncertainty)
        epistemic_var = np.var(all_means, axis=0)
        epistemic = np.sqrt(epistemic_var)
        
        # Aleatoric: mean of variances (data uncertainty)
        if len(all_vars) > 0:
            all_vars = np.array(all_vars)
            aleatoric_var = np.mean(all_vars, axis=0)
            aleatoric = np.sqrt(aleatoric_var)
        else:
            aleatoric = np.zeros_like(mean)
        
        return mean, epistemic, aleatoric
    
    def predict_with_total_uncertainty(self, X: np.ndarray
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction with total uncertainty (epistemic + aleatoric).
        
        Total variance = Var(means) + Mean(variances)
        """
        mean, epistemic, aleatoric = self.predict_with_decomposition(X)
        total_std = np.sqrt(epistemic**2 + aleatoric**2)
        return mean, total_std
    
    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output (average across ensemble)."""
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        predictions = []
        for model in self.models:
            y_lf = model.predict_lf(X)
            predictions.append(y_lf)
        
        return np.mean(predictions, axis=0)
    
    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble on test data."""
        Y_test = np.asarray(Y_test).reshape(-1, 1)
        mean, total_std = self.predict_with_total_uncertainty(X_test)
        
        y_true = Y_test.flatten()
        y_pred = mean.flatten()
        y_std = total_std.flatten()
        
        # Metrics
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # NLL
        y_std = np.maximum(y_std, 1e-6)
        nll = np.mean(0.5 * np.log(2 * np.pi * y_std**2) +
                     0.5 * ((y_true - y_pred) / y_std)**2)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nll': nll,
            'mean_std': np.mean(y_std)
        }


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing Deep Ensemble...")
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Add parent to path for imports
    import sys
    sys.path.insert(0, '..')
    
    np.random.seed(42)
    
    # Synthetic data
    X_lf = np.random.rand(50, 2).astype(np.float32)
    Y_lf = np.sin(2 * np.pi * X_lf[:, 0:1]).astype(np.float32)
    
    X_hf = np.random.rand(10, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)
    
    # Simple model factory for testing
    from models.mf_dnn import MFDNN
    
    factory = lambda: MFDNN(
        layers_lf=[2, 10, 1],
        layers_hf_nl=[3, 10, 1],
        layers_hf_l=[3, 1],
        max_epochs=1000,
        patience=100,
        verbose=False
    )
    
    # Create and train ensemble
    ensemble = DeepEnsemble(factory, n_models=3)
    info = ensemble.fit(X_lf, Y_lf, X_hf, Y_hf, verbose=True)
    
    print(f"\nEnsemble trained: {info['n_models']} models")
    
    # Predict with uncertainty
    X_test = np.random.rand(5, 2).astype(np.float32)
    mean, epistemic, aleatoric = ensemble.predict_with_decomposition(X_test)
    
    print(f"\nPredictions: {mean.flatten()}")
    print(f"Epistemic uncertainty: {epistemic.flatten()}")
    print(f"Aleatoric uncertainty: {aleatoric.flatten()}")
    
    print("\n✓ Deep Ensemble test passed!")