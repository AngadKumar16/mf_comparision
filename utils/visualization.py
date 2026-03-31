"""
Visualization Functions for Multi-Fidelity Models

This module provides:
1. Prediction surface plots (2D contour)
2. LOO scatter plots
3. Comparison bar charts
4. Calibration curves
5. Noise ablation plots

Extracted from your plotting code in Documents 2, 3, 6.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple, Any
import warnings


def plot_prediction_surface(model, 
                            X_lf: np.ndarray, 
                            X_hf: np.ndarray,
                            Y_hf: np.ndarray,
                            x_range: Tuple[float, float] = None,
                            y_range: Tuple[float, float] = None,
                            n_grid: int = 100,
                            cmap: str = 'rainbow',
                            vmin: float = None,
                            vmax: float = None,
                            title: str = "MF Prediction",
                            save_path: str = None,
                            show_lf_points: bool = True,
                            show_hf_points: bool = True):
    """
    Plot 2D prediction surface with data points.
    
    Args:
        model: Trained model with predict(X) method
        X_lf: LF input locations for scatter
        X_hf: HF input locations for scatter
        Y_hf: HF output values (for colorbar range)
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        n_grid: Grid resolution
        cmap: Colormap
        vmin, vmax: Color range
        title: Plot title
        save_path: Optional path to save figure
        show_lf_points: Show LF data points
        show_hf_points: Show HF data points
    """
    # Determine ranges from data if not provided
    if x_range is None:
        x_range = (X_lf[:, 0].min() - 0.05, X_lf[:, 0].max() + 0.05)
    if y_range is None:
        y_range = (X_lf[:, 1].min() - 0.05, X_lf[:, 1].max() + 0.05)
    
    # Create grid
    x_lin = np.linspace(x_range[0], x_range[1], n_grid)
    y_lin = np.linspace(y_range[0], y_range[1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    XY_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Predict on grid
    Z_pred, Z_std = model.predict(XY_grid, return_std=True)
    Z_pred = Z_pred.reshape(n_grid, n_grid)
    if Z_std is not None:
        Z_std = Z_std.reshape(n_grid, n_grid)
    
    # Color range
    if vmin is None:
        vmin = min(Z_pred.min(), Y_hf.min()) if Y_hf is not None else Z_pred.min()
    if vmax is None:
        vmax = max(Z_pred.max(), Y_hf.max()) if Y_hf is not None else Z_pred.max()
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if Z_std is not None else 1, 
                             figsize=(12 if Z_std is not None else 6, 5))
    
    if Z_std is None:
        axes = [axes]
    
    # Mean prediction
    ax = axes[0]
    im = ax.contourf(X_grid, Y_grid, Z_pred, levels=20, cmap=cmap, 
                     vmin=vmin, vmax=vmax)
    
    if show_lf_points:
        ax.scatter(X_lf[:, 0], X_lf[:, 1], c='gray', s=3, alpha=0.3, label='LF')
    if show_hf_points and X_hf is not None:
        ax.scatter(X_hf[:, 0], X_hf[:, 1], c='white', s=80, 
                  edgecolors='black', linewidth=1.5, marker='o', label='HF')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title} - Mean Prediction')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Output')
    
    # Uncertainty (if available)
    if Z_std is not None and len(axes) > 1:
        ax = axes[1]
        im = ax.contourf(X_grid, Y_grid, Z_std, levels=20, cmap='viridis')
        
        if show_hf_points and X_hf is not None:
            ax.scatter(X_hf[:, 0], X_hf[:, 1], c='white', s=80,
                      edgecolors='black', linewidth=1.5, marker='o')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - Uncertainty (Std)')
        plt.colorbar(im, ax=ax, label='Std Dev')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_loo_scatter(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_std: np.ndarray = None,
                     model_name: str = "Model",
                     save_path: str = None):
    """
    Plot LOO-CV true vs predicted scatter.
    
    This is your LOO validation plot from Documents 3 and 6.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Compute RMSE and R²
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Scatter with optional error bars
    if y_std is not None and not np.all(np.isnan(y_std)):
        y_std = np.asarray(y_std).flatten()
        ax.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='o',
                   markersize=10, capsize=4, alpha=0.7,
                   color='blue', ecolor='gray', label='Predictions ± 95% CI')
    else:
        ax.scatter(y_true, y_pred, c='blue', s=100, alpha=0.7, 
                  edgecolors='black', label='Predictions')
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
           max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
    
    # Annotate points
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        ax.annotate(f'{i+1}', (true, pred), textcoords="offset points",
                   xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('True Value', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'{model_name} LOO-CV\nRMSE = {rmse:.4f}, R² = {r2:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_model_comparison_bars(results: Dict[str, Dict[str, float]],
                               metrics: List[str] = None,
                               save_path: str = None):
    """
    Bar chart comparing multiple models.
    
    Args:
        results: Dict of {model_name: {metric: value}}
        metrics: List of metrics to plot (default: RMSE, MAE, R²)
        save_path: Optional path to save figure
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2']
    
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    x = np.arange(n_models)
    
    titles = {
        'rmse': 'RMSE (lower is better)',
        'mae': 'MAE (lower is better)',
        'r2': 'R² (higher is better)',
        'nll': 'NLL (lower is better)',
        'coverage_90': '90% CI Coverage'
    }
    
    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(titles.get(metric, metric))
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_calibration_curve(results: Dict[str, Dict[str, Any]],
                           save_path: str = None):
    """
    Plot reliability diagram for uncertainty calibration.
    
    Args:
        results: Dict of {model_name: {expected_coverage, observed_coverage}}
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, metrics), color in zip(results.items(), colors):
        expected = metrics.get('expected_coverage', [])
        observed = metrics.get('observed_coverage', [])
        cal_error = metrics.get('calibration_error', np.nan)
        
        if len(expected) > 0 and len(observed) > 0:
            ax.plot(expected, observed, 'o-', color=color, markersize=8,
                   label=f'{name} (ECE={cal_error:.3f})')
    
    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Observed Coverage', fontsize=12)
    ax.set_title('Uncertainty Calibration', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_noise_ablation(results: Dict[str, Dict[float, Dict[str, float]]],
                        metric: str = 'rmse',
                        save_path: str = None):
    """
    Plot RMSE vs noise level for multiple models.
    
    Args:
        results: Dict of {model_name: {noise_level: {metric: value}}}
        metric: Which metric to plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, noise_data), color in zip(results.items(), colors):
        noise_levels = sorted(noise_data.keys())
        means = [noise_data[n].get(f'{metric}_mean', noise_data[n].get(metric, 0)) 
                for n in noise_levels]
        stds = [noise_data[n].get(f'{metric}_std', 0) for n in noise_levels]
        
        ax.errorbar([n * 100 for n in noise_levels], means, yerr=stds,
                   marker='o', markersize=8, capsize=5, label=name, color=color)
    
    ax.set_xlabel('Noise Level (%)', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_multi_surface_comparison(models: Dict[str, Any],
                                   X_lf: np.ndarray,
                                   X_hf: np.ndarray,
                                   Y_hf: np.ndarray,
                                   n_grid: int = 100,
                                   cmap: str = 'rainbow',
                                   save_path: str = None):
    """
    Plot prediction surfaces for multiple models side by side.
    
    Args:
        models: Dict of {model_name: trained_model}
        X_lf, X_hf, Y_hf: Data for plotting
    """
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # Grid
    x_range = (X_lf[:, 0].min() - 0.05, X_lf[:, 0].max() + 0.05)
    y_range = (X_lf[:, 1].min() - 0.05, X_lf[:, 1].max() + 0.05)
    x_lin = np.linspace(x_range[0], x_range[1], n_grid)
    y_lin = np.linspace(y_range[0], y_range[1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    XY_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Color range (global for all plots)
    vmin = Y_hf.min()
    vmax = Y_hf.max()
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Predict
        Z_pred, _ = model.predict(XY_grid, return_std=False)
        Z_pred = Z_pred.reshape(n_grid, n_grid)
        
        # Update color range
        vmin = min(vmin, Z_pred.min())
        vmax = max(vmax, Z_pred.max())
    
    # Now plot with unified color range
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        Z_pred, _ = model.predict(XY_grid, return_std=False)
        Z_pred = Z_pred.reshape(n_grid, n_grid)
        
        im = ax.contourf(X_grid, Y_grid, Z_pred, levels=20, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.scatter(X_hf[:, 0], X_hf[:, 1], c='white', s=60,
                  edgecolors='black', linewidth=1.5, marker='o')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    # Hide unused axes
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)
    
    # Shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Output')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing visualization module...")
    
    # Create synthetic data
    np.random.seed(42)
    y_true = np.array([20, 21, 22, 23, 24])
    y_pred = np.array([20.5, 21.2, 21.8, 23.1, 23.8])
    y_std = np.array([0.3, 0.4, 0.3, 0.2, 0.4])
    
    # Test LOO scatter
    print("\nTesting LOO scatter plot...")
    plot_loo_scatter(y_true, y_pred, y_std, model_name="Test Model")
    
    # Test comparison bars
    print("\nTesting comparison bar chart...")
    results = {
        'Model A': {'rmse': 0.5, 'mae': 0.4, 'r2': 0.9},
        'Model B': {'rmse': 0.6, 'mae': 0.5, 'r2': 0.85},
        'Model C': {'rmse': 0.4, 'mae': 0.3, 'r2': 0.95},
    }
    plot_model_comparison_bars(results)
    
    print("\n✓ Visualization tests passed!")