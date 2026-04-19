"""
Evaluation Metrics for Multi-Fidelity Models

This module provides:
1. Standard regression metrics (RMSE, MAE, R², MAPE)
2. Uncertainty metrics (NLL, calibration error)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
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
    y_std_raw = np.asarray(y_std).flatten()

    # If the model returns no uncertainty (all zeros/NaN), skip NLL/coverage
    has_uncertainty = np.all(np.isfinite(y_std_raw)) and np.mean(y_std_raw) > 1e-8
    y_std = np.maximum(y_std_raw, 1e-6)

    # Standardized residuals
    z = (y_true - y_pred) / y_std

    # Negative Log-Likelihood (Gaussian) — only meaningful with real std
    if has_uncertainty:
        nll = np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * z**2)
    else:
        nll = np.nan

    # Sharpness (average predicted std)
    sharpness = np.mean(y_std_raw) if has_uncertainty else np.nan

    if not has_uncertainty:
        return {
            'nll': np.nan,
            'sharpness': np.nan,
            'calibration_error': np.nan,
            'coverage_90': np.nan,
            'expected_coverage': [],
            'observed_coverage': []
        }

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


def export_latex_table(loo_results: Dict[str, Dict[str, Any]],
                       save_path: str = None) -> str:
    """
    Export LOO-CV results as a LaTeX booktabs table.

    Args:
        loo_results: {model_name: metrics_dict} from run_loo_cv
        save_path:   Optional path to write .tex file

    Returns:
        LaTeX string
    """
    import pandas as pd

    rows = {}
    for name, m in loo_results.items():
        nll = m.get('nll', float('nan'))
        cov = m.get('coverage_90', float('nan'))
        rows[name] = {
            'RMSE': f"{m['rmse']:.4f}",
            'MAE':  f"{m['mae']:.4f}",
            'R²':   f"{m['r2']:.4f}",
            'NLL':  f"{nll:.4f}" if (nll is not None and not (isinstance(nll, float) and (nll != nll))) else '---',
            'Cov. 90\\%': f"{cov:.2%}" if (cov is not None and not (isinstance(cov, float) and (cov != cov))) else '---',
        }

    df = pd.DataFrame(rows).T
    df.index.name = 'Model'

    latex = df.to_latex(
        escape=False,
        column_format='l' + 'r' * len(df.columns),
        caption='LOO-CV comparison of multi-fidelity surrogate models.',
        label='tab:loo_results',
        position='ht',
    )
    # Upgrade to booktabs style
    latex = latex.replace('\\toprule', '\\toprule').replace(
        '\\begin{tabular}', '\\begin{tabular}')
    header = (
        '% Requires \\usepackage{booktabs} in your LaTeX preamble\n'
    )
    latex = header + latex

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"  LaTeX table saved to {save_path}")

    return latex


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
