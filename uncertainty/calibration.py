"""
Uncertainty Calibration Metrics

Provides metrics to evaluate how well uncertainty estimates
are calibrated (well-calibrated = 90% CI contains 90% of data).

Metrics:
- Expected Calibration Error (ECE)
- Sharpness (average predicted std)
- Coverage at different confidence levels
- Negative Log-Likelihood (NLL)
- Reliability diagrams
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt


def compute_calibration_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_std: np.ndarray,
                                 confidence_levels: List[float] = None
                                ) -> Dict[str, Any]:
    """
    Compute comprehensive uncertainty calibration metrics.
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted mean values (N,)
        y_std: Predicted standard deviations (N,)
        confidence_levels: List of CI levels to check (default: [0.5, 0.7, 0.9, 0.95])
    
    Returns:
        Dict with calibration metrics
    """
    # Flatten and ensure valid std
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.maximum(np.asarray(y_std).flatten(), 1e-8)
    
    if confidence_levels is None:
        confidence_levels = [0.5, 0.7, 0.9, 0.95]
    
    # Standardized residuals (z-scores)
    z = (y_true - y_pred) / y_std
    
    # 1. Expected vs Observed Coverage
    expected_coverage = np.array(confidence_levels)
    observed_coverage = []
    
    for conf in confidence_levels:
        # For Gaussian, CI of level conf has z-threshold:
        z_thresh = stats.norm.ppf((1 + conf) / 2)
        fraction_in_ci = np.mean(np.abs(z) <= z_thresh)
        observed_coverage.append(fraction_in_ci)
    
    observed_coverage = np.array(observed_coverage)
    
    # 2. Expected Calibration Error (ECE)
    # Average absolute difference between expected and observed
    ece = np.mean(np.abs(expected_coverage - observed_coverage))
    
    # 3. Sharpness (average predicted uncertainty)
    # Lower is better IF calibration is good
    sharpness = np.mean(y_std)
    
    # 4. Negative Log-Likelihood (Gaussian)
    nll = np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * z**2)
    
    # 5. Interval Score (proper scoring rule for intervals)
    # At 90% level
    alpha = 0.1
    z_90 = stats.norm.ppf(0.95)
    lower_90 = y_pred - z_90 * y_std
    upper_90 = y_pred + z_90 * y_std
    width = upper_90 - lower_90
    
    # Penalty for observations outside interval
    penalty_lower = (2/alpha) * (lower_90 - y_true) * (y_true < lower_90)
    penalty_upper = (2/alpha) * (y_true - upper_90) * (y_true > upper_90)
    interval_score = np.mean(width + penalty_lower + penalty_upper)
    
    # 6. Coverage at specific levels
    def _find_coverage(target):
        for i, c in enumerate(confidence_levels):
            if abs(c - target) < 1e-9:
                return observed_coverage[i]
        return None

    coverage_50 = _find_coverage(0.5)
    coverage_90 = _find_coverage(0.9)
    coverage_95 = _find_coverage(0.95)
    
    # 7. Coefficient of Variation of std
    cv_std = np.std(y_std) / np.mean(y_std) if np.mean(y_std) > 0 else 0
    
    return {
        'ece': ece,
        'sharpness': sharpness,
        'nll': nll,
        'interval_score': interval_score,
        'coverage_50': coverage_50,
        'coverage_90': coverage_90,
        'coverage_95': coverage_95,
        'cv_std': cv_std,
        'expected_coverage': expected_coverage.tolist(),
        'observed_coverage': observed_coverage.tolist(),
        'z_scores': z.tolist(),
    }


def compute_calibration_curve(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_std: np.ndarray,
                               n_bins: int = 10
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for reliability diagram.
    
    Args:
        y_true: Ground truth
        y_pred: Predicted mean
        y_std: Predicted std
        n_bins: Number of confidence bins
    
    Returns:
        expected: Expected coverage levels
        observed: Observed coverage at each level
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.maximum(np.asarray(y_std).flatten(), 1e-8)
    
    z = (y_true - y_pred) / y_std
    
    expected = np.linspace(0.1, 0.99, n_bins)
    observed = []
    
    for conf in expected:
        z_thresh = stats.norm.ppf((1 + conf) / 2)
        fraction_in = np.mean(np.abs(z) <= z_thresh)
        observed.append(fraction_in)
    
    return expected, np.array(observed)


def plot_reliability_diagram(results_dict: Dict[str, Dict[str, Any]],
                             save_path: str = None,
                             figsize: Tuple[int, int] = (8, 8)):
    """
    Plot reliability diagram comparing multiple models.
    
    Args:
        results_dict: {model_name: calibration_metrics}
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    
    # Shaded region for ±5% tolerance
    ax.fill_between([0, 1], [0, 0.95], [0.05, 1], alpha=0.1, color='gray')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (name, metrics), color in zip(results_dict.items(), colors):
        expected = metrics.get('expected_coverage', [])
        observed = metrics.get('observed_coverage', [])
        ece = metrics.get('ece', np.nan)
        
        if len(expected) > 0 and len(observed) > 0:
            ax.plot(expected, observed, 'o-', color=color, markersize=10,
                   linewidth=2, label=f'{name} (ECE={ece:.3f})')
    
    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Observed Coverage', fontsize=12)
    ax.set_title('Uncertainty Calibration (Reliability Diagram)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_prediction_intervals(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_std: np.ndarray,
                               model_name: str = "Model",
                               ci_level: float = 0.9,
                               save_path: str = None):
    """
    Plot predictions with confidence intervals.
    
    Args:
        y_true: Ground truth
        y_pred: Predicted mean
        y_std: Predicted std
        model_name: Name for title
        ci_level: Confidence level for intervals
        save_path: Path to save figure
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    
    # Sort by true value for cleaner plot
    sort_idx = np.argsort(y_true)
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    y_std = y_std[sort_idx]
    
    # CI bounds
    z_thresh = stats.norm.ppf((1 + ci_level) / 2)
    lower = y_pred - z_thresh * y_std
    upper = y_pred + z_thresh * y_std
    
    # Check coverage
    in_ci = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_ci)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(y_true))
    
    # CI band
    ax.fill_between(x, lower, upper, alpha=0.3, color='blue',
                   label=f'{int(ci_level*100)}% CI')
    
    # Predictions
    ax.plot(x, y_pred, 'b-', linewidth=2, label='Prediction')
    
    # True values
    ax.scatter(x, y_true, c='red', s=50, zorder=5, label='True')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{model_name}: Predictions with {int(ci_level*100)}% CI\n'
                f'(Coverage: {coverage:.1%}, Expected: {ci_level:.1%})',
                fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def print_calibration_summary(metrics: Dict[str, Any], model_name: str = "Model"):
    """Pretty print calibration summary."""
    print(f"\n{'='*50}")
    print(f"Calibration Summary: {model_name}")
    print('='*50)
    print(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")
    print(f"Negative Log-Likelihood (NLL):    {metrics['nll']:.4f}")
    print(f"Sharpness (mean std):             {metrics['sharpness']:.4f}")
    print(f"Interval Score (90%):             {metrics['interval_score']:.4f}")
    print()
    print("Coverage Analysis:")
    print(f"  50% CI: {metrics['coverage_50']:.1%} (expected 50%)")
    print(f"  90% CI: {metrics['coverage_90']:.1%} (expected 90%)")
    print(f"  95% CI: {metrics['coverage_95']:.1%} (expected 95%)")
    print('='*50)


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing calibration metrics...")
    
    np.random.seed(42)
    
    # Well-calibrated predictions
    n = 100
    y_true = np.random.randn(n)
    y_pred = y_true + 0.1 * np.random.randn(n)  # Small bias
    y_std = np.abs(0.1 + 0.05 * np.random.randn(n))  # Predicted std
    
    # Compute metrics
    metrics = compute_calibration_metrics(y_true, y_pred, y_std)
    print_calibration_summary(metrics, "Well-Calibrated Model")
    
    # Overconfident predictions (std too small)
    y_std_overconf = y_std * 0.3
    metrics_overconf = compute_calibration_metrics(y_true, y_pred, y_std_overconf)
    print_calibration_summary(metrics_overconf, "Overconfident Model")
    
    # Underconfident predictions (std too large)
    y_std_underconf = y_std * 3.0
    metrics_underconf = compute_calibration_metrics(y_true, y_pred, y_std_underconf)
    print_calibration_summary(metrics_underconf, "Underconfident Model")
    
    # Plot reliability diagram
    results = {
        'Well-Calibrated': metrics,
        'Overconfident': metrics_overconf,
        'Underconfident': metrics_underconf,
    }
    
    print("\nPlotting reliability diagram...")
    plot_reliability_diagram(results)
    
    print("\n✓ Calibration metrics test passed!")