"""
Data Loading and Preprocessing for Multi-Fidelity Models

This module handles:
1. Loading MATLAB .mat files with LF and HF data
2. Train/test splitting
3. Normalization (z-score)
4. Outlier detection and handling

Extracted from your existing code in Documents 1 and 4.
"""

import numpy as np
import scipy.io as sio
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class NormalizationStats:
    """Store normalization statistics for later denormalization."""
    mean: np.ndarray
    std: np.ndarray
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
    
    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        return X_norm * self.std + self.mean


@dataclass
class MFDataset:
    """
    Container for multi-fidelity dataset.
    
    Attributes:
        X_lf, Y_lf: Low-fidelity inputs and outputs (raw)
        X_hf_train, Y_hf_train: High-fidelity training data (raw)
        X_hf_test, Y_hf_test: High-fidelity test data (raw)
        X_lf_n, Y_lf_n: Normalized LF data
        X_hf_train_n, Y_hf_train_n: Normalized HF training data
        X_hf_test_n, Y_hf_test_n: Normalized HF test data
        norm_X_lf, norm_Y_lf: LF normalization stats
        norm_X_hf, norm_Y_hf: HF normalization stats
    """
    # Raw data
    X_lf: np.ndarray
    Y_lf: np.ndarray
    X_hf_train: np.ndarray
    Y_hf_train: np.ndarray
    X_hf_test: np.ndarray
    Y_hf_test: np.ndarray
    
    # Normalized data
    X_lf_n: np.ndarray
    Y_lf_n: np.ndarray
    X_hf_train_n: np.ndarray
    Y_hf_train_n: np.ndarray
    X_hf_test_n: np.ndarray
    Y_hf_test_n: np.ndarray
    
    # Normalization statistics
    norm_X_lf: NormalizationStats
    norm_Y_lf: NormalizationStats
    norm_X_hf: NormalizationStats
    norm_Y_hf: NormalizationStats
    
    def denormalize_Y_hf(self, Y_norm: np.ndarray) -> np.ndarray:
        """Denormalize HF outputs."""
        return self.norm_Y_hf.denormalize(Y_norm)
    
    def denormalize_Y_lf(self, Y_norm: np.ndarray) -> np.ndarray:
        """Denormalize LF outputs."""
        return self.norm_Y_lf.denormalize(Y_norm)
    
    def summary(self) -> str:
        """Print dataset summary."""
        return f"""
Multi-Fidelity Dataset Summary:
===============================
Low-Fidelity:  {self.X_lf.shape[0]} points, {self.X_lf.shape[1]}D inputs
High-Fidelity Train: {self.X_hf_train.shape[0]} points
High-Fidelity Test:  {self.X_hf_test.shape[0]} points

Y_lf range:  [{self.Y_lf.min():.2f}, {self.Y_lf.max():.2f}]
Y_hf range:  [{self.Y_hf_train.min():.2f}, {self.Y_hf_train.max():.2f}]
"""


def load_matlab_data(mat_path: str, 
                     n_hf_train: int = 12,
                     eps: float = 1e-8,
                     detect_outliers: bool = True,
                     outlier_threshold: float = 3.0) -> MFDataset:
    """
    Load multi-fidelity data from MATLAB .mat file.
    
    This is your existing data loading code from Documents 1 and 4,
    organized into a clean function.
    
    Args:
        mat_path: Path to .mat file
        n_hf_train: Number of HF points to use for training
        eps: Small value to prevent division by zero
        detect_outliers: Whether to detect and report outliers
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        MFDataset object with all data and normalization stats
    """
    # Load MATLAB data
    data = sio.loadmat(mat_path)
    LF = data['LF']
    HF = data['HF']
    
    # =========================================================
    # Extract Low-Fidelity data
    # =========================================================
    x_lf = LF['X'][0, 0][:, 0].astype(np.float64).ravel()
    y_lf = LF['X'][0, 0][:, 1].astype(np.float64).ravel()
    T_lf = LF['Y'][0, 0].astype(np.float64).ravel()
    
    X_L = np.column_stack([x_lf, y_lf])
    Y_L = T_lf[:, None]
    
    # =========================================================
    # Extract High-Fidelity data
    # =========================================================
    x_hf = HF['X'][0, 0][:, 0].astype(np.float64).ravel()
    y_hf = HF['X'][0, 0][:, 1].astype(np.float64).ravel()
    T_hf = HF['Y'][0, 0].astype(np.float64).ravel()
    
    X_H_full = np.column_stack([x_hf, y_hf])
    Y_H_full = T_hf[:, None]
    
    # =========================================================
    # Train/Test Split
    # =========================================================
    n_hf = X_H_full.shape[0]
    train_idx = np.arange(0, n_hf_train)
    test_idx = np.arange(n_hf_train, n_hf)
    
    X_H_train, Y_H_train = X_H_full[train_idx], Y_H_full[train_idx]
    X_H_test, Y_H_test = X_H_full[test_idx], Y_H_full[test_idx]
    Y_H_train = Y_H_train.reshape(-1, 1)
    
    # =========================================================
    # Compute Normalization Statistics
    # =========================================================
    # LF stats (from all LF data)
    norm_X_lf = NormalizationStats(
        mean=X_L.mean(axis=0),
        std=np.maximum(X_L.std(axis=0), eps)
    )
    norm_Y_lf = NormalizationStats(
        mean=Y_L.mean(axis=0),
        std=np.maximum(Y_L.std(axis=0), eps)
    )
    
    # HF stats (from training data only - important!)
    norm_X_hf = NormalizationStats(
        mean=X_H_train.mean(axis=0),
        std=np.maximum(X_H_train.std(axis=0), eps)
    )
    norm_Y_hf = NormalizationStats(
        mean=Y_H_train.mean(axis=0),
        std=np.maximum(Y_H_train.std(axis=0), eps)
    )
    
    # =========================================================
    # Normalize Data
    # =========================================================
    X_L_n = norm_X_lf.normalize(X_L)
    Y_L_n = norm_Y_lf.normalize(Y_L).reshape(-1, 1)
    
    X_H_train_n = norm_X_hf.normalize(X_H_train).reshape(-1, 2)
    Y_H_train_n = norm_Y_hf.normalize(Y_H_train).reshape(-1, 1)
    
    X_H_test_n = norm_X_hf.normalize(X_H_test)
    Y_H_test_n = norm_Y_hf.normalize(Y_H_test)
    
    # =========================================================
    # Outlier Detection (optional)
    # =========================================================
    if detect_outliers:
        z = (Y_H_train_n - np.mean(Y_H_train_n)) / (np.std(Y_H_train_n) + 1e-12)
        outlier_mask = np.abs(z).flatten() > outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) > 0:
            print(f"WARNING: Found {len(outlier_indices)} outliers at indices: {outlier_indices}")
            print(f"  Values: {Y_H_train[outlier_indices].flatten()}")
            print(f"  Z-scores: {z[outlier_indices].flatten()}")
    
    # =========================================================
    # Create and Return Dataset
    # =========================================================
    return MFDataset(
        X_lf=X_L, Y_lf=Y_L,
        X_hf_train=X_H_train, Y_hf_train=Y_H_train,
        X_hf_test=X_H_test, Y_hf_test=Y_H_test,
        X_lf_n=X_L_n, Y_lf_n=Y_L_n,
        X_hf_train_n=X_H_train_n, Y_hf_train_n=Y_H_train_n,
        X_hf_test_n=X_H_test_n, Y_hf_test_n=Y_H_test_n,
        norm_X_lf=norm_X_lf, norm_Y_lf=norm_Y_lf,
        norm_X_hf=norm_X_hf, norm_Y_hf=norm_Y_hf,
    )


def add_noise(Y: np.ndarray, 
              noise_percent: float = 0.10,
              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise as percentage of signal std.
    
    Args:
        Y: Target values
        noise_percent: Noise level as fraction of std (e.g., 0.10 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        Y_noisy: Noisy targets
        noise: The noise that was added
    """
    if seed is not None:
        np.random.seed(seed)
    
    std = np.std(Y)
    noise = noise_percent * std * np.random.randn(*Y.shape)
    
    return Y + noise, noise


def get_lf_predictions_at_hf(X_lf: np.ndarray, 
                              Y_lf: np.ndarray, 
                              X_hf: np.ndarray) -> np.ndarray:
    """
    Get LF predictions at HF locations using nearest neighbor interpolation.
    
    This is used for augmenting HF inputs with LF information in DNN/KAN models.
    
    Args:
        X_lf: LF input locations
        Y_lf: LF output values
        X_hf: HF input locations where we need LF predictions
    
    Returns:
        Y_lf_at_hf: LF predictions at HF locations
    """
    from scipy.interpolate import NearestNDInterpolator
    
    lf_interp = NearestNDInterpolator(X_lf, Y_lf.flatten())
    Y_lf_at_hf = lf_interp(X_hf).reshape(-1, 1)
    
    return Y_lf_at_hf


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    # Test with your data path
    import sys
    sys.path.append('..')
    from config import MATLAB_DATA_PATH, N_HF_TRAIN
    
    print("Testing data loader...")
    
    try:
        dataset = load_matlab_data(MATLAB_DATA_PATH, n_hf_train=N_HF_TRAIN)
        print(dataset.summary())
        print("✓ Data loading successful!")
    except FileNotFoundError:
        print(f"Data file not found at: {MATLAB_DATA_PATH}")
        print("Update MATLAB_DATA_PATH in config.py")