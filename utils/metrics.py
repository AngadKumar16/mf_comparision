"""
Evaluation Metrics for Multi-Fidelity Models

This module provides:
1. Standard regression metrics (RMSE, MAE, R², MAPE)
2. Uncertainty metrics (NLL, calibration error)
3. LOO-CV implementation

Extracted from your evaluation code in Documents 2, 3, 6.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dict with RMSE, MAE, R², MAPE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle zero values)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def compute_uncertainty_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_std: np.ndarray) -> Dict[str, float]:
    """
    Compute uncertainty quantification metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
    
    Returns:
        Dict with NLL, calibration error, sharpness, coverage
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.maximum(np.asarray(y_std).flatten(), 1e-6)
    
    # Standardized residuals
    z = (y_true - y_pred) / y_std
    
    # Negative Log-Likelihood (Gaussian)
    nll = np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * z**2)
    
    # Sharpness (average predicted std)
    sharpness = np.mean(y_std)
    
    # Calibration: expected vs observed coverage
    from scipy import stats
    expected_coverage = np.array([0.5, 0.7, 0.9, 0.95])
    observed_coverage = []
    
    for q in expected_coverage:
        z_threshold = stats.norm.ppf((1 + q) / 2)
        fraction_in_ci = np.mean(np.abs(z) <= z_threshold)
        observed_coverage.append(fraction_in_ci)
    
    observed_coverage = np.array(observed_coverage)
    calibration_error = np.mean(np.abs(expected_coverage - observed_coverage))
    
    # 90% CI coverage
    z_90 = stats.norm.ppf(0.95)
    coverage_90 = np.mean(np.abs(z) <= z_90)
    
    return {
        'nll': nll,
        'sharpness': sharpness,
        'calibration_error': calibration_error,
        'coverage_90': coverage_90,
        'expected_coverage': expected_coverage.tolist(),
        'observed_coverage': observed_coverage.tolist()
    }


def run_loo_cv(model_factory: Callable,
               X_lf: np.ndarray, Y_lf: np.ndarray,
               X_hf: np.ndarray, Y_hf: np.ndarray,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Run Leave-One-Out Cross-Validation on HF points.
    
    This is your LOO-CV code from Documents 2, 3, 6 unified into one function.
    
    Args:
        model_factory: Function that returns a new model instance
        X_lf: LF input data
        Y_lf: LF output data
        X_hf: HF input data (will be split in LOO fashion)
        Y_hf: HF output data
        verbose: Print progress
    
    Returns:
        Dict with predictions, errors, and metrics
    """
    X_lf = np.asarray(X_lf)
    Y_lf = np.asarray(Y_lf).reshape(-1, 1)
    X_hf = np.asarray(X_hf)
    Y_hf = np.asarray(Y_hf).reshape(-1, 1)
    
    n_hf = X_hf.shape[0]
    loo = LeaveOneOut()
    
    # Storage
    y_true_list = []
    y_pred_list = []
    y_std_list = []
    fold_errors = []
    
    if verbose:
        print(f"Running LOO-CV with {n_hf} folds...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(X_hf)):
        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_hf}", end='\r')
        
        # Split HF data
        X_hf_train = X_hf[train_idx]
        Y_hf_train = Y_hf[train_idx]
        X_hf_val = X_hf[val_idx]
        Y_hf_val = Y_hf[val_idx]
        
        # Create and train model
        model = model_factory()
        model.fit(X_lf, Y_lf, X_hf_train, Y_hf_train)
        
        # Predict
        y_pred, y_std = model.predict(X_hf_val, return_std=True)
        
        # Store results
        y_true_list.append(Y_hf_val.flatten()[0])
        y_pred_list.append(y_pred.flatten()[0])
        if y_std is not None:
            y_std_list.append(y_std.flatten()[0])
        else:
            y_std_list.append(np.nan)
        
        error = Y_hf_val.flatten()[0] - y_pred.flatten()[0]
        fold_errors.append(error)
    
    if verbose:
        print(f"  Completed {n_hf} folds.          ")
    
    # Convert to arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_std = np.array(y_std_list)
    
    # Compute metrics
    regression_metrics = compute_regression_metrics(y_true, y_pred)
    
    # Uncertainty metrics (if std available)
    if not np.all(np.isnan(y_std)):
        uncertainty_metrics = compute_uncertainty_metrics(y_true, y_pred, y_std)
    else:
        uncertainty_metrics = {
            'nll': np.nan,
            'calibration_error': np.nan,
            'coverage_90': np.nan
        }
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_std': y_std,
        'fold_errors': np.array(fold_errors),
        **regression_metrics,
        **uncertainty_metrics
    }


def print_metrics_summary(metrics: Dict[str, Any], model_name: str = "Model"):
    """Pretty print metrics summary."""
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print('='*50)
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"R²:    {metrics['r2']:.4f}")
    if not np.isnan(metrics.get('nll', np.nan)):
        print(f"NLL:   {metrics['nll']:.4f}")
        print(f"Coverage (90%): {metrics['coverage_90']:.2%}")
    print('='*50)


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing metrics module...")
    
    # Synthetic data
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    y_std = np.array([0.2, 0.3, 0.2, 0.1, 0.3])
    
    # Test regression metrics
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    print(f"\nRegression metrics: {reg_metrics}")
    
    # Test uncertainty metrics
    unc_metrics = compute_uncertainty_metrics(y_true, y_pred, y_std)
    print(f"Uncertainty metrics: {unc_metrics}")
    
    print("\n✓ Metrics tests passed!")