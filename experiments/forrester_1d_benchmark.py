"""
1D Forrester Benchmark — Literature Reproduction Test

Goal: Determine if the MF code reproduces published results on the canonical
1D Forrester benchmark, OR if there's a bug causing systematic underperformance.

Setup (matching Meng & Karniadakis 2020, Section 4.1):
    HF function: f_h(x) = (6x - 2)^2 * sin(12x - 4)
    LF function: f_l(x) = 0.5 * f_h(x) + 10*(x - 0.5) - 5
    Domain:      x in [0, 1]
    n_LF:        50 points (sampled uniformly or LHS)
    n_HF train:  4 points (Meng paper uses 4 for sparse regime)
    n_HF test:   500 points on a dense grid (for stable RMSE)

Expected results (from Meng & Karniadakis 2020, Fig. 5):
    MF-GP:  RMSE ≈ 0.01-0.05 (near-perfect on this benchmark)
    MF-DNN: RMSE ≈ 0.05-0.20 (slightly worse, depending on n_HF)

If our DNN gets RMSE > 1.0 on this setup, there's a real bug.
If our DNN gets RMSE 0.05-0.30, the code is correct and our 2D Forrester
results just reflect the harder benchmark.

Run from project root:
    python experiments/forrester_1d_benchmark.py
"""

import sys
import os
os.environ['TF_DISABLE_METAL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import NormalizingModelWrapper
from utils.metrics import compute_regression_metrics

from models.mf_gp import MFGP_Linear
from models.mf_dnn import MFDNN
from models.mf_kan import MFKAN
from models.mf_hybrid import HybridKANDNN


# ──────────────────────────────────────────────────────────────────────────────
# 1D Forrester functions (canonical literature definitions)
# ──────────────────────────────────────────────────────────────────────────────
def forrester_hf_1d(X):
    """Standard 1D Forrester HF function. Input: (N, 1). Output: (N, 1)."""
    X = np.atleast_2d(X)
    if X.shape[1] != 1:
        X = X.reshape(-1, 1)
    x = X[:, 0]
    return ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).reshape(-1, 1)


def forrester_lf_1d(X):
    """Standard 1D Forrester LF function. Input: (N, 1). Output: (N, 1)."""
    X = np.atleast_2d(X)
    if X.shape[1] != 1:
        X = X.reshape(-1, 1)
    x = X[:, 0]
    hf = forrester_hf_1d(X).flatten()
    return (0.5 * hf + 10 * (x - 0.5) - 5).reshape(-1, 1)


def generate_data(n_lf=50, n_hf_train=4, n_hf_test=500, seed=42):
    """Generate 1D Forrester train/test data."""
    rng = np.random.default_rng(seed)

    # LF: latin-hypercube-style spread (1D = uniform spacing with jitter)
    X_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    X_lf += rng.uniform(-0.5/n_lf, 0.5/n_lf, X_lf.shape)
    X_lf = np.clip(X_lf, 0, 1).astype(np.float32)

    # HF train: well-spread small set
    X_hf_train = np.linspace(0, 1, n_hf_train).reshape(-1, 1).astype(np.float32)

    # HF test: dense grid
    X_hf_test = np.linspace(0, 1, n_hf_test).reshape(-1, 1).astype(np.float32)

    Y_lf = forrester_lf_1d(X_lf).astype(np.float32)
    Y_hf_train = forrester_hf_1d(X_hf_train).astype(np.float32)
    Y_hf_test = forrester_hf_1d(X_hf_test).astype(np.float32)

    return {
        'X_lf': X_lf, 'Y_lf': Y_lf,
        'X_hf_train': X_hf_train, 'Y_hf_train': Y_hf_train,
        'X_hf_test': X_hf_test, 'Y_hf_test': Y_hf_test,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model factories — using simpler architectures matching the 1D problem
# ──────────────────────────────────────────────────────────────────────────────
def make_factories():
    """Architectures sized for 1D problem (input dim = 1)."""
    return {
        'GP-Linear': lambda: MFGP_Linear(num_restarts=10),

        'DNN': lambda: MFDNN(
            layers_lf=[1, 20, 20, 1],
            layers_hf_nl=[2, 10, 10, 1],   # input = [x, y_lf_pred]
            layers_hf_l=[2, 1],
            learning_rate=0.001,
            max_epochs=10000,
            patience=2000,
            l2_reg=0.01,
            lf_pretrain_patience=500,
            verbose=False,
        ),

        'KAN': lambda: MFKAN(
            layers_lf=[1, 20, 20, 1],
            layers_hf_nl=[2, 10, 10, 1],
            layers_hf_l=[2, 1],
            grid_size=3,
            spline_order=3,
            learning_rate=0.001,
            max_epochs=10000,
            patience=2000,
            lf_pretrain_patience=500,
            verbose=False,
        ),

        'Hybrid': lambda: HybridKANDNN(
            layers_lf=[1, 20, 20, 1],
            layers_hf_nl=[2, 10, 10, 1],
            layers_hf_l=[2, 1],
            grid_size=3,
            spline_order=3,
            learning_rate=0.001,
            max_epochs=10000,
            patience=2000,
            l2_reg=0.01,
            lf_pretrain_patience=500,
            verbose=False,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single training pass — train on (X_lf, Y_lf, X_hf_train, Y_hf_train),
# evaluate on (X_hf_test, Y_hf_test). Repeat with seeds for variance.
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(name, factory, data, n_seeds=3):
    """Train/eval n_seeds times, report mean ± std RMSE."""
    rmses = []
    maes  = []
    times = []

    for seed in range(n_seeds):
        np.random.seed(42 + seed)
        tf.random.set_seed(42 + seed)

        t0 = time.time()
        model = NormalizingModelWrapper(factory())
        model.fit(data['X_lf'], data['Y_lf'],
                  data['X_hf_train'], data['Y_hf_train'])
        y_pred, _ = model.predict(data['X_hf_test'], return_std=False)
        elapsed = time.time() - t0

        m = compute_regression_metrics(data['Y_hf_test'], y_pred)
        rmses.append(m['rmse'])
        maes.append(m['mae'])
        times.append(elapsed)

    rmses = np.array(rmses)
    maes  = np.array(maes)
    print(f"  {name:<10} RMSE = {rmses.mean():.4f} ± {rmses.std():.4f}   "
          f"MAE = {maes.mean():.4f}   time/run = {np.mean(times):.1f}s")
    return {'name': name, 'rmse_mean': rmses.mean(), 'rmse_std': rmses.std(),
            'mae_mean': maes.mean(), 'rmses': rmses.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic: print data ranges, LF-HF correlation
# ──────────────────────────────────────────────────────────────────────────────
def print_diagnostics(data):
    Y_lf = data['Y_lf'].flatten()
    Y_hf_train = data['Y_hf_train'].flatten()
    Y_hf_test = data['Y_hf_test'].flatten()

    print(f"  n_LF = {len(Y_lf)}, n_HF_train = {len(Y_hf_train)}, "
          f"n_HF_test = {len(Y_hf_test)}")
    print(f"  Y_lf range:       [{Y_lf.min():.3f}, {Y_lf.max():.3f}]")
    print(f"  Y_hf train range: [{Y_hf_train.min():.3f}, {Y_hf_train.max():.3f}]")
    print(f"  Y_hf test range:  [{Y_hf_test.min():.3f}, {Y_hf_test.max():.3f}]")
    print(f"  Y_hf test std:    {Y_hf_test.std():.3f} "
          "(this is the 'predict-the-mean' RMSE baseline)")

    # LF vs HF at the LF locations
    Y_hf_at_lf = forrester_hf_1d(data['X_lf']).flatten()
    rho = np.corrcoef(Y_lf, Y_hf_at_lf)[0, 1]
    print(f"  Pearson ρ(Y_lf, Y_hf at LF locations): {rho:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("1D FORRESTER BENCHMARK — LITERATURE REPRODUCTION TEST")
    print("=" * 70)
    print()
    print("Goal: Verify our MF code reproduces published results on the")
    print("canonical 1D Forrester benchmark.")
    print()
    print("Reference: Meng & Karniadakis (2020), Section 4.1")
    print("Expected MF-GP RMSE: ~0.01-0.05")
    print("Expected MF-DNN RMSE: ~0.05-0.30")
    print()

    # Test multiple training set sizes (sparse to dense)
    for n_hf in [4, 8, 14]:
        print()
        print("=" * 70)
        print(f"n_HF_train = {n_hf}")
        print("=" * 70)

        data = generate_data(n_lf=50, n_hf_train=n_hf, n_hf_test=500)
        print_diagnostics(data)
        print()

        factories = make_factories()
        for name, factory in factories.items():
            evaluate(name, factory, data, n_seeds=3)

    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("If MF-DNN RMSE at n_HF=14 is below 0.30: code is correct, no bug.")
    print("If MF-DNN RMSE at n_HF=14 is above 1.00: there's likely a bug.")
    print()
    print("If GP-Linear RMSE is consistently < 0.05: GP code working as expected.")
    print("If GP-Linear RMSE is > 0.5: GP code may have an issue.")


if __name__ == "__main__":
    main()