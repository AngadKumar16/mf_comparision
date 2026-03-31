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
                
                # Train model
                model = factory()
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
                   noise_results: dict = None, save: bool = True):
    """Generate all visualization plots."""
    from utils.visualization import (
        plot_loo_scatter, 
        plot_model_comparison_bars,
        plot_noise_ablation
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
    
    # 3. Noise ablation (if available)
    if noise_results:
        save_path = FIGURES_DIR / 'noise_ablation.png' if save else None
        plot_noise_ablation(noise_results, metric='rmse', save_path=save_path)


def save_results_csv(loo_results: dict, noise_results: dict = None):
    """Save results to CSV files."""
    import pandas as pd
    
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


def main(use_synthetic: bool = False, 
         run_noise: bool = False,
         save_plots: bool = True):
    """
    Main experiment entry point.
    
    Args:
        use_synthetic: Use synthetic Forrester data for debugging
        run_noise: Run noise ablation study
        save_plots: Save plots to files
    """
    print("="*60)
    print("MULTI-FIDELITY MODEL COMPARISON")
    print("="*60)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    data = load_data(use_synthetic=use_synthetic)
    
    # 2. Create model factories
    print("\n[2/4] Initializing models...")
    model_factories = create_model_factories()
    print(f"  Models: {list(model_factories.keys())}")
    
    # 3. Run LOO-CV
    print("\n[3/4] Running LOO-CV...")
    loo_results = run_loo_comparison(data, model_factories, verbose=True)
    
    # 4. Optional: Noise ablation
    noise_results = None
    if run_noise:
        print("\n[4/4] Running noise ablation...")
        noise_results = run_noise_ablation(
            data, model_factories, 
            noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20],
            n_trials=3
        )
    else:
        print("\n[4/4] Skipping noise ablation (use --noise to enable)")
    
    # 5. Generate plots
    print("\n[5/5] Generating plots...")
    generate_plots(data, loo_results, noise_results, save=save_plots)
    
    # 6. Save results
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MF model comparison')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic Forrester data for debugging')
    parser.add_argument('--noise', action='store_true',
                       help='Run noise ablation study')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots')
    
    args = parser.parse_args()
    
    main(
        use_synthetic=args.synthetic,
        run_noise=args.noise,
        save_plots=not args.no_save
    )