"""
Main Experiment Script for Multi-Fidelity Model Comparison

This script:
1. Loads data (real or synthetic)
2. Trains GP, DNN, KAN models
3. Runs LOO-CV
4. Generates comparison plots and tables

Run from project root:
    python experiments/run_comparison.py
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MATLAB_DATA_PATH, N_HF_TRAIN, RANDOM_SEED,
    GP_CONFIG, DNN_CONFIG, KAN_CONFIG,
    NOISE_LEVELS, N_NOISE_TRIALS,
    RESULTS_DIR, FIGURES_DIR
)
from utils.data_utils import NormalizingModelWrapper


def load_data(use_synthetic: bool = False):
    """Load data from MATLAB file or generate synthetic."""
    if use_synthetic:
        print("Using synthetic Forrester data...")
        from data.synthetic.forrester import Forrester2D
        data = Forrester2D.generate_data()
        return data
    else:
        print(f"Loading real data from {MATLAB_DATA_PATH}...")
        from data.loader import load_matlab_data
        dataset = load_matlab_data(MATLAB_DATA_PATH, n_hf_train=N_HF_TRAIN)
        print(dataset.summary())
        return {
            'X_lf': dataset.X_lf,
            'Y_lf': dataset.Y_lf,
            'X_hf_train': dataset.X_hf_train,
            'Y_hf_train': dataset.Y_hf_train,
            'X_hf_test': dataset.X_hf_test,
            'Y_hf_test': dataset.Y_hf_test,
            'X_lf_n': dataset.X_lf_n,
            'Y_lf_n': dataset.Y_lf_n,
            'X_hf_train_n': dataset.X_hf_train_n,
            'Y_hf_train_n': dataset.Y_hf_train_n,
            'X_hf_test_n': dataset.X_hf_test_n,
            'Y_hf_test_n': dataset.Y_hf_test_n,
            'dataset': dataset,
        }


def create_model_factories():
    """Create model factory functions."""
    from models.mf_gp import MFGP_Linear
    from models.mf_dnn import MFDNN
    from models.mf_kan import MFKAN
    
    factories = {
        'GP-Linear': lambda: MFGP_Linear(num_restarts=GP_CONFIG['num_restarts']),
        
        'DNN': lambda: MFDNN(
            layers_lf=DNN_CONFIG['layers_lf'],
            layers_hf_nl=DNN_CONFIG['layers_hf_nl'],
            layers_hf_l=DNN_CONFIG['layers_hf_l'],
            learning_rate=DNN_CONFIG['learning_rate'],
            max_epochs=DNN_CONFIG['max_epochs'],
            patience=DNN_CONFIG['patience'],
            lf_pretrain_patience=DNN_CONFIG.get('lf_pretrain_patience', 500),
            verbose=False
        ),

        'KAN': lambda: MFKAN(
            layers_lf=KAN_CONFIG['layers_lf'],
            layers_hf_nl=KAN_CONFIG['layers_hf_nl'],
            layers_hf_l=KAN_CONFIG['layers_hf_l'],
            grid_size=KAN_CONFIG['grid_size'],
            spline_order=KAN_CONFIG['spline_order'],
            learning_rate=KAN_CONFIG['learning_rate'],
            max_epochs=KAN_CONFIG['max_epochs'],
            patience=KAN_CONFIG['patience'],
            lf_pretrain_patience=KAN_CONFIG.get('lf_pretrain_patience', 500),
            verbose=False
        ),
    }
    
    return factories


def run_loo_comparison(data: dict, model_factories: dict, verbose: bool = True):
    """Run LOO-CV for all models and compare."""
    from utils.metrics import run_loo_cv, print_metrics_summary
    
    results = {}
    
    for name, factory in model_factories.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running LOO-CV for {name}")
            print('='*50)
        
        metrics = run_loo_cv(
            model_factory=factory,
            X_lf=data['X_lf'],
            Y_lf=data['Y_lf'],
            X_hf=data['X_hf_train'],
            Y_hf=data['Y_hf_train'],
            verbose=verbose
        )
        
        results[name] = metrics
        
        if verbose:
            print_metrics_summary(metrics, name)
    
    return results


def run_noise_ablation(data: dict, model_factories: dict, 
                       noise_levels: list = None, n_trials: int = 3):
    """Run noise ablation study."""
    from data.loader import add_noise
    from utils.metrics import compute_regression_metrics
    
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    
    results = {name: {} for name in model_factories.keys()}
    
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Noise Level: {noise_level*100:.0f}%")
        print('='*50)
        
        for name, factory in model_factories.items():
            trial_metrics = []
            
            for trial in range(n_trials):
                # Add noise
                Y_hf_noisy, _ = add_noise(
                    data['Y_hf_train'], 
                    noise_level, 
                    seed=RANDOM_SEED + trial * 1000
                )
                
                # Train model (wrapper handles normalization)
                model = NormalizingModelWrapper(factory())
                model.fit(data['X_lf'], data['Y_lf'],
                         data['X_hf_train'], Y_hf_noisy)

                # Evaluate on clean test data
                y_pred, _ = model.predict(data['X_hf_test'], return_std=False)
                metrics = compute_regression_metrics(data['Y_hf_test'], y_pred)
                trial_metrics.append(metrics)
            
            # Average across trials
            avg_metrics = {}
            for key in trial_metrics[0].keys():
                values = [m[key] for m in trial_metrics]
                avg_metrics[f'{key}_mean'] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            
            results[name][noise_level] = avg_metrics
            print(f"  {name}: RMSE = {avg_metrics['rmse_mean']:.4f} ± {avg_metrics['rmse_std']:.4f}")
    
    return results


def generate_plots(data: dict, loo_results: dict,
                   noise_results: dict = None, save: bool = True,
                   training_histories: dict = None,
                   trained_models: dict = None):
    """Generate all visualization plots."""
    from utils.visualization import (
        plot_loo_scatter,
        plot_model_comparison_bars,
        plot_noise_ablation,
        plot_calibration_curve,
        plot_residual_analysis,
        plot_lf_hf_scatter,
        plot_training_convergence,
    )

    # 1. LOO scatter plots for each model
    for name, metrics in loo_results.items():
        save_path = FIGURES_DIR / f'loo_{name.lower().replace("-", "_")}.png' if save else None
        plot_loo_scatter(
            metrics['y_true'],
            metrics['y_pred'],
            metrics['y_std'],
            model_name=name,
            save_path=save_path
        )

    # 2. Comparison bar chart
    comparison_data = {name: {
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2']
    } for name, metrics in loo_results.items()}

    save_path = FIGURES_DIR / 'model_comparison.png' if save else None
    plot_model_comparison_bars(comparison_data, save_path=save_path)

    # 3. Residual diagnostics (histograms, boxplots, Q-Q)
    save_path = FIGURES_DIR / 'residual_analysis.png' if save else None
    plot_residual_analysis(loo_results, save_path=save_path)

    # 4. LF–HF relationship scatter
    gp_model = (trained_models or {}).get('GP-Linear', None)
    save_path = FIGURES_DIR / 'lf_hf_correlation.png' if save else None
    plot_lf_hf_scatter(
        data['X_lf'], data['Y_lf'],
        data['X_hf_train'], data['Y_hf_train'],
        gp_model=gp_model,
        save_path=save_path
    )

    # 5. Calibration curves (models that provide std)
    calib_data = {
        name: metrics for name, metrics in loo_results.items()
        if metrics.get('expected_coverage') and len(metrics['expected_coverage']) > 0
    }
    if calib_data:
        save_path = FIGURES_DIR / 'calibration_curves.png' if save else None
        plot_calibration_curve(calib_data, save_path=save_path)

    # 6. Training convergence curves (DNN / KAN)
    if training_histories:
        save_path = FIGURES_DIR / 'training_convergence.png' if save else None
        plot_training_convergence(training_histories, save_path=save_path)

    # 7. Noise ablation (if available)
    if noise_results:
        save_path = FIGURES_DIR / 'noise_ablation.png' if save else None
        plot_noise_ablation(noise_results, metric='rmse', save_path=save_path)


def save_results_csv(loo_results: dict, noise_results: dict = None):
    """Save results to CSV files and export LaTeX table."""
    import pandas as pd
    from utils.metrics import export_latex_table

    # LOO results
    loo_df = pd.DataFrame({
        name: {
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'NLL': metrics.get('nll', np.nan),
            'Coverage 90%': metrics.get('coverage_90', np.nan)
        }
        for name, metrics in loo_results.items()
    }).T

    loo_df.to_csv(RESULTS_DIR / 'loo_results.csv')
    print(f"\nSaved LOO results to {RESULTS_DIR / 'loo_results.csv'}")

    # LaTeX table
    export_latex_table(loo_results, save_path=str(RESULTS_DIR / 'table_loo.tex'))
    
    # Noise ablation
    if noise_results:
        rows = []
        for model, noise_data in noise_results.items():
            for noise_level, metrics in noise_data.items():
                rows.append({
                    'Model': model,
                    'Noise Level': noise_level,
                    'RMSE Mean': metrics['rmse_mean'],
                    'RMSE Std': metrics['rmse_std'],
                    'MAE Mean': metrics['mae_mean'],
                    'MAE Std': metrics['mae_std'],
                })
        
        noise_df = pd.DataFrame(rows)
        noise_df.to_csv(RESULTS_DIR / 'noise_ablation.csv', index=False)
        print(f"Saved noise ablation to {RESULTS_DIR / 'noise_ablation.csv'}")


def train_all_models(data: dict, model_factories: dict) -> tuple:
    """Train one instance of each model on the full HF training set.

    Returns:
        trained:           {name: model}
        training_histories: {name: history_list}  (NN models only)
    """
    from uncertainty.ensemble import DeepEnsemble

    trained = {}
    training_histories = {}

    for name, factory in model_factories.items():
        print(f"  Training {name}...")
        m = NormalizingModelWrapper(factory())
        info = m.fit(data['X_lf'], data['Y_lf'],
                     data['X_hf_train'], data['Y_hf_train'])
        trained[name] = m
        if info and 'history' in info:
            training_histories[name] = info['history']

    # Build ensemble wrappers for DNN and KAN to provide uncertainty maps.
    # The single-model versions remain for LOO-CV (speed); ensembles are only
    # used during PyGMT / uncertainty visualisation.
    print("\n  Building Deep Ensembles (n=5) for DNN & KAN uncertainty maps...")
    for name in ('DNN', 'KAN'):
        if name not in model_factories:
            continue
        factory = model_factories[name]
        wrapped_factory = lambda f=factory: NormalizingModelWrapper(f())
        ens = DeepEnsemble(wrapped_factory, n_models=5)
        ens.fit(data['X_lf'], data['Y_lf'],
                data['X_hf_train'], data['Y_hf_train'],
                verbose=False)
        trained[f'{name}-Ensemble'] = ens

    return trained, training_histories


def run_pygmt_maps(data: dict, trained_models: dict,
                   loo_results: dict = None,
                   figures_dir=None):
    """Generate PyGMT maps for all models."""
    try:
        from utils.pygmt_maps import generate_all_maps
    except ImportError as e:
        print(f"  PyGMT not available ({e}) — skipping maps.")
        return

    from config import PLOT_CONFIG
    vmin, vmax = PLOT_CONFIG.get('temp_range', (None, None))
    out = str(figures_dir / "pygmt") if figures_dir else "figures/pygmt"

    generate_all_maps(
        models=trained_models,
        X_lf=data['X_lf'],
        Y_lf=data['Y_lf'],
        X_hf=data['X_hf_train'],
        Y_hf=data['Y_hf_train'],
        loo_results=loo_results,
        figures_dir=out,
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
    )


def main(use_synthetic: bool = False,
         run_noise: bool = False,
         save_plots: bool = True,
         run_pygmt: bool = True):
    """
    Main experiment entry point.

    Args:
        use_synthetic: Use synthetic Forrester data for debugging
        run_noise: Run noise ablation study
        save_plots: Save matplotlib plots to files
        run_pygmt: Generate PyGMT individual maps
    """
    print("="*60)
    print("MULTI-FIDELITY MODEL COMPARISON")
    print("="*60)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    print("\n[1/5] Loading data...")
    data = load_data(use_synthetic=use_synthetic)

    # 2. Create model factories and train full models once
    print("\n[2/5] Initializing and training models on full HF set...")
    model_factories = create_model_factories()
    print(f"  Models: {list(model_factories.keys())}")
    trained_models, training_histories = train_all_models(data, model_factories)

    # 3. Run LOO-CV (reuses factories, not the trained models)
    print("\n[3/5] Running LOO-CV...")
    loo_results = run_loo_comparison(data, model_factories, verbose=True)

    # 4. Optional: Noise ablation
    noise_results = None
    if run_noise:
        print("\n[4/5] Running noise ablation...")
        noise_results = run_noise_ablation(
            data, model_factories,
            noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20],
            n_trials=3
        )
    else:
        print("\n[4/5] Skipping noise ablation (use --noise to enable)")

    # 5. Generate matplotlib plots
    print("\n[5/5] Generating matplotlib plots...")
    generate_plots(data, loo_results, noise_results, save=save_plots,
                   training_histories=training_histories,
                   trained_models=trained_models)

    # 6. PyGMT maps (reuse already-trained models — no extra training pass)
    # Pass ensemble versions for DNN/KAN so uncertainty maps are populated.
    if run_pygmt and save_plots:
        print("\n[+] Generating PyGMT maps...")
        pygmt_models = {
            name: trained_models.get(f'{name}-Ensemble', trained_models[name])
            for name in model_factories
            if name in trained_models
        }
        # Ensure GP-Linear uses its single-model version (already has UQ)
        pygmt_models['GP-Linear'] = trained_models['GP-Linear']
        run_pygmt_maps(data, pygmt_models, loo_results,
                       figures_dir=FIGURES_DIR)

    # 7. Save results
    save_results_csv(loo_results, noise_results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nLOO-CV Results:")
    print("-"*40)
    for name, metrics in loo_results.items():
        print(f"  {name:15s}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    if noise_results:
        print("\nNoise Robustness (RMSE at 10% noise):")
        print("-"*40)
        for name in noise_results.keys():
            if 0.10 in noise_results[name]:
                rmse = noise_results[name][0.10]['rmse_mean']
                print(f"  {name:15s}: {rmse:.4f}")

    print("\n✓ Experiment complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  PyGMT maps:       {FIGURES_DIR / 'pygmt'}")


def _loo_scalars(loo_results: dict) -> dict:
    """Strip array columns from LOO results, keep only scalar metrics."""
    scalar_keys = ('rmse', 'mae', 'r2', 'nll', 'coverage_90',
                   'mse', 'mape', 'sharpness', 'calibration_error')
    return {
        model: {k: v for k, v in m.items() if k in scalar_keys}
        for model, m in loo_results.items()
    }


def save_scenario_csv(scenario_results: dict, save_path: str):
    """Write unified scenario comparison CSV ({scenario: {model: metrics}})."""
    import pandas as pd

    rows = []
    for scenario, models in scenario_results.items():
        for model, m in models.items():
            rows.append({
                'Scenario':      scenario,
                'Model':         model,
                'RMSE':          m.get('rmse',        np.nan),
                'MAE':           m.get('mae',         np.nan),
                'R²':            m.get('r2',          np.nan),
                'NLL':           m.get('nll',         np.nan),
                'Coverage 90%':  m.get('coverage_90', np.nan),
            })

    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"  Scenario CSV   → {save_path}")


def export_scenario_latex(scenario_results: dict, save_path: str):
    """Write a LaTeX booktabs table with Scenario+Model as index."""
    import pandas as pd

    rows = []
    for scenario, models in scenario_results.items():
        for model, m in models.items():
            nll = m.get('nll', np.nan)
            cov = m.get('coverage_90', np.nan)
            rows.append({
                'Scenario': scenario,
                'Model':    model,
                'RMSE':     f"{m.get('rmse', np.nan):.4f}",
                'MAE':      f"{m.get('mae',  np.nan):.4f}",
                'R\u00b2':  f"{m.get('r2',   np.nan):.4f}",
                'NLL':      f"{nll:.4f}" if np.isfinite(nll) else '---',
                'Cov.\\ 90\\%': f"{cov:.2%}" if np.isfinite(cov) else '---',
            })

    df = pd.DataFrame(rows).set_index(['Scenario', 'Model'])
    latex = (
        '% Requires \\usepackage{booktabs,multirow}\n'
        + df.to_latex(
            escape=False,
            column_format='ll' + 'r' * len(df.columns),
            caption='LOO-CV metrics across all comparison scenarios.',
            label='tab:scenario_comparison',
            position='ht',
            multirow=True,
        )
    )
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Scenario LaTeX → {save_path}")


def run_all_scenarios(run_noise: bool = True, save: bool = True):
    """
    Run Forrester + Real (clean) LOO-CV, optional noise ablation, and
    produce a unified cross-scenario comparison figure and table.

    Called via:  python experiments/run_comparison.py --all-scenarios [--noise]
    """
    from utils.visualization import plot_scenario_comparison

    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("CROSS-SCENARIO MODEL COMPARISON")
    print("=" * 60)

    scenario_results = {}
    noise_results    = None

    # ── 1. Forrester (synthetic) ──────────────────────────────────────
    print("\n[1/3] Forrester (synthetic) — LOO-CV...")
    forr_data = load_data(use_synthetic=True)
    forr_fac  = create_model_factories()
    forr_loo  = run_loo_comparison(forr_data, forr_fac, verbose=True)
    scenario_results['Forrester'] = _loo_scalars(forr_loo)

    # ── 2. Real data, clean ───────────────────────────────────────────
    print("\n[2/3] Real data (clean) — LOO-CV...")
    real_data = load_data(use_synthetic=False)
    real_fac  = create_model_factories()
    real_loo  = run_loo_comparison(real_data, real_fac, verbose=True)
    scenario_results['Real (clean)'] = _loo_scalars(real_loo)

    # ── 3. Noise ablation on real data ────────────────────────────────
    if run_noise:
        print("\n[3/3] Real data — noise ablation...")
        noise_results = run_noise_ablation(
            real_data, real_fac,
            noise_levels=NOISE_LEVELS,
            n_trials=N_NOISE_TRIALS,
        )
    else:
        print("\n[3/3] Skipping noise ablation (add --noise to enable)")

    # ── Save outputs ──────────────────────────────────────────────────
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        print("\nSaving outputs...")
        save_scenario_csv(scenario_results,
                          save_path=str(RESULTS_DIR / 'scenario_comparison.csv'))
        export_scenario_latex(scenario_results,
                              save_path=str(RESULTS_DIR / 'table_scenarios.tex'))

    # ── Figure ────────────────────────────────────────────────────────
    print("\nGenerating cross-scenario figure...")
    fig_path = str(FIGURES_DIR / 'scenario_comparison.png') if save else None
    plot_scenario_comparison(
        scenario_results=scenario_results,
        noise_results=noise_results,
        metrics=['rmse', 'mae', 'r2'],
        save_path=fig_path,
    )
    if fig_path:
        print(f"  Figure         → {fig_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hdr = f"  {'Scenario':<18} {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for scenario, models in scenario_results.items():
        for model, m in models.items():
            print(f"  {scenario:<18} {model:<12}"
                  f" {m.get('rmse', np.nan):>8.4f}"
                  f" {m.get('mae',  np.nan):>8.4f}"
                  f" {m.get('r2',   np.nan):>8.4f}")

    print("\n✓ Done.")
    if save:
        print(f"  Results: {RESULTS_DIR}")
        print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run MF model comparison')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic Forrester data for debugging')
    parser.add_argument('--noise', action='store_true',
                        help='Run noise ablation study')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots')
    parser.add_argument('--no-pygmt', action='store_true',
                        help='Skip PyGMT map generation')
    parser.add_argument('--all-scenarios', action='store_true',
                        help='Run Forrester + Real + optional noise and compare')

    args = parser.parse_args()

    if args.all_scenarios:
        run_all_scenarios(
            run_noise=args.noise,
            save=not args.no_save,
        )
    else:
        main(
            use_synthetic=args.synthetic,
            run_noise=args.noise,
            save_plots=not args.no_save,
            run_pygmt=not args.no_pygmt,
        )