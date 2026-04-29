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
os.environ['TF_DISABLE_METAL'] = '1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import warnings
from sklearn.model_selection import LeaveOneOut
from utils.metrics import (compute_regression_metrics,
                                compute_uncertainty_metrics,
                                print_metrics_summary)
from uncertainty.ensemble import DeepEnsemble
import time

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MATLAB_DATA_PATH, N_HF_TRAIN, RANDOM_SEED,
    GP_CONFIG, DNN_CONFIG, KAN_CONFIG, HYBRID_CONFIG,
    NOISE_LEVELS, N_NOISE_TRIALS,
    RESULTS_DIR, FIGURES_DIR
)
from utils.data_utils import NormalizingModelWrapper


def load_data(use_synthetic: bool = False):
    """Load data from MATLAB file or generate synthetic.
    Both paths return identical 12/2 train/test splits."""
    if use_synthetic:
        print("Using synthetic Forrester data...")
        from data.synthetic.forrester import Forrester2D
        data = Forrester2D.generate_data(n_lf=200, n_hf_train=14)
        # Split 12 train / 2 test to match AGU 2025 protocol on real data
        X_hf_full = data['X_hf_train']
        Y_hf_full = data['Y_hf_train']
        data['X_hf_train'] = X_hf_full[:12]
        data['Y_hf_train'] = Y_hf_full[:12]
        data['X_hf_test']  = X_hf_full[12:14]
        data['Y_hf_test']  = Y_hf_full[12:14]
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
    from models.mf_hybrid import HybridKANDNN

    factories = {
        'GP-Linear': lambda: MFGP_Linear(num_restarts=GP_CONFIG['num_restarts']),

        'DNN': lambda: MFDNN(
            layers_lf=DNN_CONFIG['layers_lf'],
            layers_hf_nl=DNN_CONFIG['layers_hf_nl'],
            layers_hf_l=DNN_CONFIG['layers_hf_l'],
            learning_rate=DNN_CONFIG['learning_rate'],
            max_epochs=DNN_CONFIG['max_epochs'],
            patience=DNN_CONFIG['patience'],
            l2_reg=DNN_CONFIG['l2_reg'],
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

        'Hybrid': lambda: HybridKANDNN(
            layers_lf=HYBRID_CONFIG['layers_lf'],
            layers_hf_nl=HYBRID_CONFIG['layers_hf_nl'],
            layers_hf_l=HYBRID_CONFIG['layers_hf_l'],
            grid_size=HYBRID_CONFIG['grid_size'],
            spline_order=HYBRID_CONFIG['spline_order'],
            learning_rate=HYBRID_CONFIG['learning_rate'],
            l2_reg=HYBRID_CONFIG['l2_reg'],
            max_epochs=HYBRID_CONFIG['max_epochs'],
            patience=HYBRID_CONFIG['patience'],
            lf_pretrain_patience=HYBRID_CONFIG['lf_pretrain_patience'],
        ),
    }

    return factories


def run_loo_cv(data: dict, model_factories: dict,
               verbose: bool = True, ensemble_nn: bool = True):
    """True leave-one-out CV across HF training points.

    For each of the n_hf training points, hold one out, train on the
    remaining n_hf-1 points (plus all LF data), and predict the held-out
    point. Aggregate predictions across folds and compute metrics.

    Args:
        data: Dict with X_lf, Y_lf, X_hf_train, Y_hf_train.
        model_factories: {name: factory_function}
        ensemble_nn: Wrap DNN/KAN/Hybrid with DeepEnsemble(n=5) for
                     meaningful uncertainty alongside GP-Linear.
    """

    X_hf = np.asarray(data['X_hf_train'])
    Y_hf = np.asarray(data['Y_hf_train']).reshape(-1, 1)
    n = len(X_hf)

    results = {}

    for name, factory in model_factories.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running LOO-CV for {name} ({n} folds)")
            print('='*50)

        effective_factory = factory
        if ensemble_nn and name in ('DNN', 'KAN', 'Hybrid'):
            effective_factory = lambda f=factory: DeepEnsemble(f, n_models=5)

        yt, yp, ys = [], [], []
        t_start = time.time()
        try:
            for fold_idx, (tr, va) in enumerate(LeaveOneOut().split(X_hf)):
                if verbose:
                    print(f"  Fold {fold_idx+1}/{n}...", end='\r')

                model = NormalizingModelWrapper(effective_factory())
                model.fit(data['X_lf'], data['Y_lf'], X_hf[tr], Y_hf[tr])

                p, s = model.predict(X_hf[va], return_std=True)
                if s is None:
                    s = np.zeros_like(p)

                yt.append(float(Y_hf[va].flatten()[0]))
                yp.append(float(p.flatten()[0]))
                ys.append(float(s.flatten()[0]))

            yt = np.array(yt)
            yp = np.array(yp)
            ys = np.array(ys)

            reg = compute_regression_metrics(yt, yp)
            unc = compute_uncertainty_metrics(yt, yp, ys)
            metrics = {
                'y_true': yt, 'y_pred': yp, 'y_std': ys,
                'n_folds': n,
                **reg, **unc,
            }

            elapsed = time.time() - t_start
            print(f"  Completed {n} folds in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"\n  !! {name} LOO-CV failed after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
            continue

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
            t_model_start = time.time()

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
            
            elapsed = time.time() - t_model_start
            results[name][noise_level] = avg_metrics
            print(f"  {name}: RMSE = {avg_metrics['rmse_mean']:.4f} ± {avg_metrics['rmse_std']:.4f}  [{elapsed:.1f}s]")
    
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
        if metrics is None:
            continue
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
    } for name, metrics in loo_results.items() if metrics is not None}

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
        if metrics is not None
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
        t_start = time.time()
        m = NormalizingModelWrapper(factory())
        info = m.fit(data['X_lf'], data['Y_lf'],
                     data['X_hf_train'], data['Y_hf_train'])
        elapsed = time.time() - t_start
        print(f"  {name} done: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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
        t_start = time.time()
        ens.fit(data['X_lf'], data['Y_lf'],
                data['X_hf_train'], data['Y_hf_train'],
                verbose=False)
        elapsed = time.time() - t_start
        print(f"  {name}-Ensemble done: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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
        if metrics is None:
            print(f"  {name:15s}: FAILED")
            continue
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
                   'mse', 'mape', 'sharpness', 'calibration_error', 'n_folds')
    return {
        model: {k: v for k, v in m.items() if k in scalar_keys}
        for model, m in loo_results.items()
        if m is not None
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

    # Extract n_folds from the first available result for caption annotation
    n_folds = None
    for models in scenario_results.values():
        for m in models.values():
            if 'n_folds' in m:
                n_folds = int(m['n_folds'])
                break
        if n_folds is not None:
            break
    n_str = f" ($n={n_folds}$ folds per dataset)" if n_folds else ""

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
            caption=f'LOO-CV metrics across all comparison scenarios{n_str}.',
            label='tab:scenario_comparison',
            position='ht',
            multirow=True,
        )
    )
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Scenario LaTeX → {save_path}")


def export_noise_latex(noise_results: dict, save_path: str):
    """Write RMSE ± std noise ablation table as LaTeX booktabs."""
    import pandas as pd

    # Collect all noise levels in sorted order
    noise_levels = sorted({nl for m in noise_results.values() for nl in m.keys()})
    col_labels = {nl: f"{int(nl*100)}\\%" for nl in noise_levels}

    rows = []
    for model, noise_data in noise_results.items():
        row = {'Model': model}
        for nl in noise_levels:
            m = noise_data.get(nl, {})
            mean = m.get('rmse_mean', np.nan)
            std  = m.get('rmse_std',  np.nan)
            row[col_labels[nl]] = (
                f"{mean:.4f} $\\pm$ {std:.4f}"
                if np.isfinite(mean) else '---'
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Model')
    latex = (
        '% Requires \\usepackage{booktabs} in your LaTeX preamble\n'
        + df.to_latex(
            escape=False,
            column_format='l' + 'r' * len(df.columns),
            caption='Noise ablation study: RMSE (mean $\\pm$ std) at each noise level.',
            label='tab:noise_ablation',
            position='ht',
        )
    )
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Noise LaTeX    → {save_path}")


def run_all_scenarios(run_noise: bool = True, save: bool = True):
    """
    Run Forrester + Real (clean) LOO-CV, optional noise ablation, and
    produce a unified cross-scenario comparison figure and table.

    Called via:  python experiments/run_comparison.py --all-scenarios [--noise]
    """
    from utils.visualization import plot_scenario_comparison

    np.random.seed(RANDOM_SEED)
    t_start = time.time()

    print("=" * 60)
    print("CROSS-SCENARIO MODEL COMPARISON")
    print("=" * 60)

    scenario_results = {}
    noise_results    = None

    # ── 1. Forrester (synthetic) ──────────────────────────────────────
    print("\n[1/3] Forrester (synthetic) — LOO-CV...")
    t1 = time.time()
    forr_data = load_data(use_synthetic=True)
    forr_fac  = create_model_factories()
    forr_loo = run_loo_comparison(forr_data, forr_fac, verbose=True)
    scenario_results['Forrester'] = _loo_scalars(forr_loo)
    print(f"  [1/3] Forrester total: {time.time()-t1:.1f}s ({(time.time()-t1)/60:.1f} min)")

    # ── 2. Real data, clean ───────────────────────────────────────────
    print("\n[2/3] Real data (clean) — LOO-CV...")
    t2 = time.time()
    real_data = load_data(use_synthetic=False)
    real_fac  = create_model_factories()
    real_loo = run_loo_comparison(real_data, real_fac, verbose=True)
    scenario_results['Real (clean)'] = _loo_scalars(real_loo)
    print(f"  [2/3] Real (clean) total: {time.time()-t2:.1f}s ({(time.time()-t2)/60:.1f} min)")

    # ── 3. Noise ablation on real data ────────────────────────────────
    if run_noise:
        print("\n[3/3] Real data — noise ablation...")
        t3 = time.time()
        noise_results = run_noise_ablation(
            real_data, real_fac,
            noise_levels=NOISE_LEVELS,
            n_trials=N_NOISE_TRIALS,
        )
        print(f"  [3/3] Noise ablation total: {time.time()-t3:.1f}s ({(time.time()-t3)/60:.1f} min)")
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

        if noise_results is not None:
            import pandas as pd
            noise_rows = []
            for model, noise_data in noise_results.items():
                for nl, m in noise_data.items():
                    noise_rows.append({
                        'Model':      model,
                        'Noise Level': nl,
                        'RMSE Mean':  m.get('rmse_mean', np.nan),
                        'RMSE Std':   m.get('rmse_std',  np.nan),
                        'MAE Mean':   m.get('mae_mean',  np.nan),
                        'MAE Std':    m.get('mae_std',   np.nan),
                    })
            pd.DataFrame(noise_rows).to_csv(
                RESULTS_DIR / 'noise_ablation.csv', index=False)
            print(f"  Noise CSV      → {RESULTS_DIR / 'noise_ablation.csv'}")
            export_noise_latex(noise_results,
                               save_path=str(RESULTS_DIR / 'table_noise_ablation.tex'))

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

    total_elapsed = time.time() - t_start
    print(f"\n✓ Done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min).")

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