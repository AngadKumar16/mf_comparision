"""
Noise Ablation Study

Systematically evaluates model robustness across different
noise levels (0%, 5%, 10%, 15%, 20%).

Generates:
- RMSE vs Noise Level plots
- LaTeX table for paper
- CSV results
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Callable
from config import MATLAB_DATA_PATH, N_HF_TRAIN, NOISE_LEVELS, N_NOISE_TRIALS, GP_CONFIG, DNN_CONFIG, KAN_CONFIG, HYBRID_CONFIG
from data.loader import load_matlab_data, add_noise
from models.mf_gp import MFGP_Linear
from models.mf_dnn import MFDNN
from models.mf_kan import MFKAN
from models.mf_hybrid import HybridKANDNN
from utils.metrics import compute_regression_metrics


class NoiseAblationStudy:
    """
    Systematic noise robustness evaluation.
    """
    
    def __init__(self,
                 model_factories: Dict[str, Callable],
                 noise_levels: List[float] = None,
                 n_trials: int = 5,
                 seed: int = 42):
        """
        Args:
            model_factories: {name: factory_function}
            noise_levels: List of noise fractions (e.g., [0, 0.05, 0.1])
            n_trials: Number of random trials per noise level
            seed: Base random seed
        """
        self.model_factories = model_factories
        self.noise_levels = noise_levels or [0.0, 0.05, 0.10, 0.15, 0.20]
        self.n_trials = n_trials
        self.seed = seed
        self.results = []
    
    def run(self, X_lf, Y_lf, X_hf_train, Y_hf_train, X_hf_test, Y_hf_test,
            verbose: bool = True) -> pd.DataFrame:
        """
        Run full ablation study.
        
        Returns:
            DataFrame with all results
        """
        self.results = []
        
        for noise_level in self.noise_levels:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Noise Level: {noise_level*100:.0f}%")
                print('='*50)
            
            for trial in range(self.n_trials):
                trial_seed = self.seed + trial * 1000
                
                # Add noise to HF training data
                Y_hf_noisy, _ = add_noise(Y_hf_train, noise_level, seed=trial_seed)
                
                for model_name, factory in self.model_factories.items():
                    if verbose:
                        print(f"  {model_name}, Trial {trial+1}/{self.n_trials}", end='\r')
                    
                    try:
                        # Train
                        model = factory()
                        model.fit(X_lf, Y_lf, X_hf_train, Y_hf_noisy)
                        
                        # Evaluate on clean test data
                        y_pred, _ = model.predict(X_hf_test, return_std=False)
                        metrics = compute_regression_metrics(Y_hf_test, y_pred)
                        
                        self.results.append({
                            'model': model_name,
                            'noise_level': noise_level,
                            'trial': trial,
                            **metrics
                        })
                    
                    except Exception as e:
                        print(f"\n  ERROR ({model_name}): {e}")
                        self.results.append({
                            'model': model_name,
                            'noise_level': noise_level,
                            'trial': trial,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'r2': np.nan,
                        })
                
                if verbose:
                    print(f"  Completed trial {trial+1}/{self.n_trials}          ")
        
        return pd.DataFrame(self.results)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics (mean ± std)."""
        df = pd.DataFrame(self.results)
        
        summary = df.groupby(['model', 'noise_level']).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std'],
        }).round(4)
        
        return summary
    
    def plot_results(self, metric: str = 'rmse', save_path: str = None):
        """Plot metric vs noise level."""
        df = pd.DataFrame(self.results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_factories)))
        
        for model_name, color in zip(self.model_factories.keys(), colors):
            model_df = df[df['model'] == model_name]
            
            grouped = model_df.groupby('noise_level')[metric]
            means = grouped.mean()
            stds = grouped.std()
            
            ax.errorbar(
                means.index * 100,
                means.values,
                yerr=stds.values,
                marker='o',
                markersize=8,
                capsize=5,
                linewidth=2,
                label=model_name,
                color=color
            )
        
        ax.set_xlabel('Noise Level (%)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Noise Level', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def generate_latex_table(self, metric: str = 'rmse') -> str:
        """Generate LaTeX table for paper."""
        df = pd.DataFrame(self.results)
        
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Noise Robustness Comparison (" + metric.upper() + ")}",
            r"\begin{tabular}{l" + "c" * len(self.noise_levels) + "}",
            r"\toprule",
        ]
        
        # Header
        header = "Model & " + " & ".join([f"{int(n*100)}\\%" for n in self.noise_levels]) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        # Data rows
        for model_name in self.model_factories.keys():
            model_df = df[df['model'] == model_name]
            row_vals = []
            
            for noise in self.noise_levels:
                subset = model_df[model_df['noise_level'] == noise]
                mean = subset[metric].mean()
                std = subset[metric].std()
                row_vals.append(f"${mean:.3f} \\pm {std:.3f}$")
            
            lines.append(f"{model_name} & " + " & ".join(row_vals) + r" \\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)


def run_noise_ablation(use_synthetic: bool = False):
    """Main function to run noise ablation."""
    
    print("="*60)
    print("NOISE ABLATION STUDY")
    print("="*60)
    
    # Load data
    if use_synthetic:
        from data.synthetic.forrester import Forrester2D
        data = Forrester2D.generate_data(n_lf=200, n_hf_train=12, n_hf_test=50)
        X_lf, Y_lf = data['X_lf'], data['Y_lf']
        X_hf_train, Y_hf_train = data['X_hf_train'], data['Y_hf_train']
        X_hf_test, Y_hf_test = data['X_hf_test'], data['Y_hf_test']
    else:
        dataset = load_matlab_data(MATLAB_DATA_PATH, n_hf_train=N_HF_TRAIN)
        X_lf, Y_lf = dataset.X_lf, dataset.Y_lf
        X_hf_train, Y_hf_train = dataset.X_hf_train, dataset.Y_hf_train
        X_hf_test, Y_hf_test = dataset.X_hf_test, dataset.Y_hf_test
    
    # Model factories
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

        'Hybrid': lambda: HybridKANDNN(
            kan_layers=HYBRID_CONFIG['kan_layers'],
            mlp_layers=HYBRID_CONFIG['mlp_layers'],
            kan_grid_size=HYBRID_CONFIG['kan_grid_size'],
            kan_spline_order=HYBRID_CONFIG['kan_spline_order'],
            dropout_rate=HYBRID_CONFIG['dropout_rate'],
            max_epochs=DNN_CONFIG['max_epochs'],
            patience=DNN_CONFIG['patience'],
            verbose=False
        ),
    }
    
    # Run study
    study = NoiseAblationStudy(
        factories,
        noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20],
        n_trials=3
    )
    
    df = study.run(X_lf, Y_lf, X_hf_train, Y_hf_train, X_hf_test, Y_hf_test)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(study.get_summary())
    
    # Plot
    study.plot_results(metric='rmse')
    
    # LaTeX table
    print("\nLaTeX Table:")
    print(study.generate_latex_table())
    
    # Save results
    df.to_csv('noise_ablation_results.csv', index=False)
    print("\nResults saved to noise_ablation_results.csv")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    
    run_noise_ablation(use_synthetic=args.synthetic)