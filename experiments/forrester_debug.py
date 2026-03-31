"""
Forrester Function Debugging Script

Use this FIRST before running on real data.
Validates all models on synthetic data with known ground truth.

Success criteria:
- All models achieve R² > 0.9 on Forrester
- Uncertainty estimates are reasonably calibrated
- No normalization/scaling bugs
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic.forrester import Forrester2D
from models.mf_gp import MFGP_Linear
from models.mf_dnn import MFDNN
from models.mf_kan import MFKAN
from models.mf_hybrid import HybridKANDNN
from utils.metrics import compute_regression_metrics
from utils.visualization import plot_loo_scatter, plot_model_comparison_bars


def run_forrester_validation(verbose: bool = True):
    """
    Validate all models on Forrester 2D function.
    """
    print("="*60)
    print("FORRESTER FUNCTION VALIDATION")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    data = Forrester2D.generate_data(
        n_lf=200,
        n_hf_train=12,
        n_hf_test=50
    )
    
    if verbose:
        print(f"\nData generated:")
        print(f"  LF points: {data['X_lf'].shape[0]}")
        print(f"  HF train:  {data['X_hf_train'].shape[0]}")
        print(f"  HF test:   {data['X_hf_test'].shape[0]}")
    
    # Define models
    models = {
        'GP-Linear': MFGP_Linear(num_restarts=3),
        
        'DNN': MFDNN(
            layers_lf=[2, 20, 20, 1],
            layers_hf_nl=[3, 20, 20, 1],
            layers_hf_l=[3, 1],
            max_epochs=10000,
            patience=500,
            verbose=False
        ),
        
        'KAN': MFKAN(
            layers_lf=[2, 20, 20, 1],
            layers_hf_nl=[3, 20, 20, 1],
            layers_hf_l=[3, 1],
            grid_size=5,
            max_epochs=10000,
            patience=500,
            verbose=False
        ),
        
        'Hybrid': HybridKANDNN(
            kan_layers=[2, 20, 20, 1],
            mlp_layers=[3, 32, 32, 1],
            max_epochs=10000,
            patience=500,
            verbose=False
        ),
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        if verbose:
            print(f"\n--- Training {name} ---")
        
        try:
            # Train
            model.fit(
                data['X_lf'], data['Y_lf'],
                data['X_hf_train'], data['Y_hf_train']
            )
            
            # Predict on test set
            y_pred, y_std = model.predict(data['X_hf_test'], return_std=True)
            
            # Compute metrics
            metrics = compute_regression_metrics(data['Y_hf_test'], y_pred)
            results[name] = metrics
            
            if verbose:
                status = "PASS" if metrics['r2'] > 0.9 else "FAIL"
                print(f"  R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f} [{status}]")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, m in results.items():
        status = "PASS" if m.get('r2', 0) > 0.9 else "FAIL"
        print(f"  {name:15s}: R²={m.get('r2', np.nan):.4f} [{status}]")
        if m.get('r2', 0) <= 0.9:
            all_passed = False
    
    if all_passed:
        print("\nAll models passed! Ready for real data.")
    else:
        print("\nSome models failed. Debug before using real data.")
    
    return results


if __name__ == "__main__":
    run_forrester_validation()