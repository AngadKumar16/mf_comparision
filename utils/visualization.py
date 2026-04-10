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


def plot_residual_analysis(loo_results: Dict[str, Dict[str, Any]],
                           save_path: str = None):
    """
    Three-panel residual diagnostic figure for LOO-CV results.

    Panels:
      Left  — Histogram of signed residuals with Gaussian overlay (per model)
      Centre — Side-by-side boxplots of signed residuals
      Right  — Q-Q plot of standardised residuals (normality check)
    """
    from scipy import stats

    model_names = list(loo_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Left: histograms ─────────────────────────────────────────────────
    ax = axes[0]
    for (name, res), color in zip(loo_results.items(), colors):
        y_true = np.asarray(res['y_true']).flatten()
        y_pred = np.asarray(res['y_pred']).flatten()
        resid = y_pred - y_true
        ax.hist(resid, bins=8, alpha=0.45, color=color, label=name, density=True)
        x_fit = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 100)
        mu, sigma = resid.mean(), resid.std()
        if sigma > 0:
            ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), color=color, lw=2)
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_xlabel('Residual (pred − true)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Centre: boxplots ─────────────────────────────────────────────────
    ax = axes[1]
    residuals = []
    for name, res in loo_results.items():
        y_true = np.asarray(res['y_true']).flatten()
        y_pred = np.asarray(res['y_pred']).flatten()
        residuals.append(y_pred - y_true)
    bp = ax.boxplot(residuals, labels=model_names, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.set_ylabel('Residual (pred − true)')
    ax.set_title('Residual Boxplots')
    ax.grid(axis='y', alpha=0.3)

    # ── Right: Q-Q plots ─────────────────────────────────────────────────
    ax = axes[2]
    for (name, res), color in zip(loo_results.items(), colors):
        y_true = np.asarray(res['y_true']).flatten()
        y_pred = np.asarray(res['y_pred']).flatten()
        resid = y_pred - y_true
        sigma = resid.std()
        z = resid / sigma if sigma > 0 else resid
        (osm, osr), _ = stats.probplot(z, dist='norm')
        ax.scatter(osm, osr, color=color, s=50, label=name, zorder=3)
    ax.plot([-3, 3], [-3, 3], 'k--', lw=1.5)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Normal Q-Q (standardised residuals)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('LOO-CV Residual Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig


def plot_lf_hf_scatter(X_lf: np.ndarray, Y_lf: np.ndarray,
                        X_hf: np.ndarray, Y_hf: np.ndarray,
                        gp_model=None,
                        save_path: str = None):
    """
    Scatter of LF values (interpolated to HF locations) vs HF truth.

    Annotates with the GP-Linear ρ (rho) scaling parameter if the model
    is provided. Shows the LF-HF linear relationship that motivates
    multi-fidelity modelling.
    """
    from scipy.spatial import cKDTree

    Y_lf_flat = np.asarray(Y_lf).flatten()
    Y_hf_flat = np.asarray(Y_hf).flatten()

    # Nearest-neighbour interpolation of LF onto HF locations
    tree = cKDTree(X_lf)
    _, idx = tree.query(X_hf)
    Y_lf_at_hf = Y_lf_flat[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y_lf_at_hf, Y_hf_flat, s=80, color='steelblue',
               edgecolors='black', linewidth=0.8, zorder=3, label='HF obs')

    # Best-fit line
    m, b = np.polyfit(Y_lf_at_hf, Y_hf_flat, 1)
    x_line = np.linspace(Y_lf_at_hf.min(), Y_lf_at_hf.max(), 100)
    ax.plot(x_line, m * x_line + b, 'r--', lw=2,
            label=f'Linear fit  (slope={m:.3f})')

    # GP ρ annotation
    if gp_model is not None and hasattr(gp_model, 'rho'):
        rho = float(np.asarray(gp_model.rho).flatten()[0])
        ax.text(0.05, 0.92, f'GP-Linear  ρ = {rho:.3f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('LF value at HF location (°C)', fontsize=12)
    ax.set_ylabel('HF observed value (°C)', fontsize=12)
    ax.set_title('LF–HF Relationship\n(nearest-neighbour LF interpolation)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig


def plot_training_convergence(training_histories: Dict[str, list],
                               save_path: str = None):
    """
    Training loss curves for neural-network models (DNN, KAN).

    Args:
        training_histories: {model_name: list of {'epoch', 'loss', 'loss_lf', 'loss_hf'}}
    """
    if not training_histories:
        return None

    fig, axes = plt.subplots(1, len(training_histories),
                              figsize=(6 * len(training_histories), 4),
                              squeeze=False)
    axes = axes[0]

    for ax, (name, history) in zip(axes, training_histories.items()):
        if not history:
            ax.set_visible(False)
            continue

        epochs = [h['epoch'] for h in history]
        loss_total = [h['loss'] for h in history]
        loss_lf = [h.get('loss_lf', np.nan) for h in history]
        loss_hf = [h.get('loss_hf', np.nan) for h in history]

        ax.plot(epochs, loss_total, lw=2, label='Total', color='black')
        if not all(np.isnan(loss_lf)):
            ax.plot(epochs, loss_lf, lw=1.5, ls='--', label='LF', color='steelblue')
        if not all(np.isnan(loss_hf)):
            ax.plot(epochs, loss_hf, lw=1.5, ls='--', label='HF', color='tomato')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'{name} Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig


def plot_scenario_comparison(
        scenario_results: Dict[str, Dict[str, Any]],
        noise_results: Dict[str, Any] = None,
        metrics: List[str] = None,
        save_path: str = None):
    """
    Two-row cross-scenario comparison figure.

    Row 1 — Grouped bar chart for each metric (RMSE, MAE, R²).
             Groups = scenarios; bars within each group = models.
    Row 2 — Noise-ablation RMSE-vs-noise-level curve (omitted if
             noise_results is None).

    Args:
        scenario_results: {scenario_name: {model_name: metrics_dict}}
        noise_results:    {model_name: {noise_level: metrics_dict}}
                          metrics_dict must have 'rmse_mean' / 'rmse_std' keys.
        metrics:          Which metrics to show in the bar chart.
        save_path:        Optional PNG path.
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2']

    has_noise = noise_results is not None and len(noise_results) > 0
    n_metric_panels = len(metrics)

    # Layout: 1 or 2 rows
    if has_noise:
        fig = plt.figure(figsize=(5 * n_metric_panels, 10))
        gs = fig.add_gridspec(2, n_metric_panels,
                              height_ratios=[1, 0.7], hspace=0.45)
        bar_axes = [fig.add_subplot(gs[0, i]) for i in range(n_metric_panels)]
        noise_ax = fig.add_subplot(gs[1, :])
    else:
        fig, bar_axes = plt.subplots(1, n_metric_panels,
                                     figsize=(5 * n_metric_panels, 5))
        if n_metric_panels == 1:
            bar_axes = [bar_axes]

    scenario_names = list(scenario_results.keys())
    # Collect model names in consistent order
    model_names = list(next(iter(scenario_results.values())).keys())
    n_scenarios = len(scenario_names)
    n_models = len(model_names)

    model_colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    bar_width = 0.8 / n_models
    x = np.arange(n_scenarios)

    metric_meta = {
        'rmse': ('RMSE', 'lower is better', False),
        'mae':  ('MAE',  'lower is better', False),
        'r2':   ('R²',   'higher is better', True),
        'nll':  ('NLL',  'lower is better', False),
    }

    # ── Row 1: grouped bar charts ─────────────────────────────────────
    for ax, metric in zip(bar_axes, metrics):
        label, direction, _ = metric_meta.get(metric, (metric.upper(), '', False))
        for m_idx, (mname, color) in enumerate(zip(model_names, model_colors)):
            vals = [
                scenario_results[s].get(mname, {}).get(metric, np.nan)
                for s in scenario_names
            ]
            offsets = x + (m_idx - (n_models - 1) / 2) * bar_width
            bars = ax.bar(offsets, vals, width=bar_width * 0.9,
                          color=color, label=mname, alpha=0.85)

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01 * abs(bar.get_height() + 1e-9),
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}\n({direction})', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    # ── Row 2: noise ablation curve ───────────────────────────────────
    if has_noise:
        for mname, color in zip(model_names, model_colors):
            if mname not in noise_results:
                continue
            noise_data = noise_results[mname]
            levels = sorted(noise_data.keys())
            means = [noise_data[n].get('rmse_mean', noise_data[n].get('rmse', np.nan))
                     for n in levels]
            stds  = [noise_data[n].get('rmse_std', 0) for n in levels]
            noise_ax.errorbar([l * 100 for l in levels], means, yerr=stds,
                              marker='o', markersize=7, capsize=4,
                              label=mname, color=color, lw=2)

        noise_ax.set_xlabel('Noise Level (%)', fontsize=11)
        noise_ax.set_ylabel('RMSE', fontsize=11)
        noise_ax.set_title('Noise Ablation — RMSE vs Noise Level (Real data)',
                           fontsize=11)
        noise_ax.legend(fontsize=9)
        noise_ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-Scenario Model Comparison', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

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