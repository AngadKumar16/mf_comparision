"""
Leave-One-Out Cross-Validation Script

Runs LOO-CV on HF points for all models and generates
comparison plots and tables.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import LeaveOneOut
from typing import Callable, Dict, Any

from config import MATLAB_DATA_PATH, N_HF_TRAIN
from data.loader import load_matlab_data
from models.mf_gp import MFGP_Linear
from models.mf_dnn import MFDNN
from models.mf_kan import MFKAN
from models.mf_hybrid import HybridKANDNN
from utils.metrics import compute_regression_metrics
from utils.visualization import plot_loo_scatter


def run_loo_cv(model_factory: Callable,
               X_lf: np.ndarray, Y_lf: np.ndarray,
               X_hf: np.ndarray, Y_hf: np.ndarray,
               model_name: str = "Model",
               verbose: bool = True) -> Dict[str, Any]:
    """
    Run Leave-One-Out CV on HF points.
    
    Args:
        model_factory: Function returning new model instance
        X_lf, Y_lf: LF data (used in full for each fold)
        X_hf, Y_hf: HF data (split via LOO)
        model_name: Name for printing
        verbose: Print progress
    
    Returns:
        Dict with predictions, errors, metrics
    """
    X_hf = np.asarray(X_hf)
    Y_hf = np.asarray(Y_hf).reshape(-1, 1)
    
    n_folds = len(X_hf)
    loo = LeaveOneOut()
    
    y_true_list = []
    y_pred_list = []
    y_std_list = []
    
    if verbose:
        print(f"\nRunning LOO-CV for {model_name} ({n_folds} folds)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(X_hf)):
        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}", end='\r')
        
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
        
        # Store
        y_true_list.append(Y_hf_val.flatten()[0])
        y_pred_list.append(y_pred.flatten()[0])
        if y_std is not None:
            y_std_list.append(y_std.flatten()[0])
        else:
            y_std_list.append(np.nan)
    
    if verbose:
        print(f"  Completed {n_folds} folds.          ")
    
    # Convert to arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_std = np.array(y_std_list)
    
    # Compute metrics
    metrics = compute_regression_metrics(y_true, y_pred)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_std': y_std,
        **metrics
    }


def run_all_loo(use_synthetic: bool = False):
    """Run LOO-CV for all models."""
    
    print("="*60)
    print("LOO-CV COMPARISON")
    print("="*60)
    
    # Load data
    if use_synthetic:
        from data.synthetic.forrester import Forrester2D
        data = Forrester2D.generate_data(n_lf=200, n_hf_train=12)
        X_lf, Y_lf = data['X_lf'], data['Y_lf']
        X_hf, Y_hf = data['X_hf_train'], data['Y_hf_train']
    else:
        dataset = load_matlab_data(MATLAB_DATA_PATH, n_hf_train=N_HF_TRAIN)
        X_lf, Y_lf = dataset.X_lf, dataset.Y_lf
        X_hf, Y_hf = dataset.X_hf_train, dataset.Y_hf_train
    
    print(f"Data: {X_lf.shape[0]} LF, {X_hf.shape[0]} HF points")
    
    # Model factories
    factories = {
        'GP-Linear': lambda: MFGP_Linear(num_restarts=5),
        
        'DNN': lambda: MFDNN(
            layers_lf=[2, 20, 20, 1],
            layers_hf_nl=[3, 20, 20, 1],
            layers_hf_l=[3, 1],
            max_epochs=200,
            patience=10,
            verbose=False
        ),
        
        'KAN': lambda: MFKAN(
            layers_lf=[2, 20, 20, 1],
            layers_hf_nl=[3, 20, 20, 1],
            layers_hf_l=[3, 1],
            max_epochs=200,
            patience=10,
            verbose=False
        ),
        
        'Hybrid': lambda: HybridKANDNN(
            kan_layers=[2, 20, 20, 1],
            mlp_layers=[3, 32, 32, 1],
            max_epochs=200,
            patience=10,
            verbose=False
        ),
    }
    
    results = {}
    
    for name, factory in factories.items():
        loo_result = run_loo_cv(
            factory, X_lf, Y_lf, X_hf, Y_hf,
            model_name=name, verbose=True
        )
        results[name] = loo_result
        
        print(f"  {name}: RMSE={loo_result['rmse']:.4f}, R²={loo_result['r2']:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    for name, res in results.items():
        plot_loo_scatter(
            res['y_true'], res['y_pred'], res['y_std'],
            model_name=name
        )
    
    # Save results
    summary = pd.DataFrame({
        name: {
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'R²': res['r2'],
        }
        for name, res in results.items()
    }).T
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary.to_string())
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    
    run_all_loo(use_synthetic=args.synthetic)