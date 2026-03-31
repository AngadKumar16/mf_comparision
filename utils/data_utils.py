"""
Data Utilities

Helper functions for data manipulation, splitting, and augmentation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import NearestNDInterpolator


def normalize(X: np.ndarray, 
              mean: np.ndarray = None, 
              std: np.ndarray = None,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalization.
    
    Args:
        X: Data to normalize
        mean: Pre-computed mean (if None, computed from X)
        std: Pre-computed std (if None, computed from X)
        eps: Small value for numerical stability
    
    Returns:
        X_norm: Normalized data
        mean: Mean used
        std: Std used
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = np.maximum(X.std(axis=0), eps)
    
    X_norm = (X - mean) / std
    return X_norm, mean, std


def denormalize(X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse z-score normalization."""
    return X_norm * std + mean


def minmax_normalize(X: np.ndarray,
                     xmin: np.ndarray = None,
                     xmax: np.ndarray = None,
                     target_range: Tuple[float, float] = (-1, 1)
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max normalization to target range.
    
    Args:
        X: Data to normalize
        xmin: Pre-computed min
        xmax: Pre-computed max
        target_range: (low, high) target range
    
    Returns:
        X_norm: Normalized data
        xmin: Min used
        xmax: Max used
    """
    if xmin is None:
        xmin = X.min(axis=0)
    if xmax is None:
        xmax = X.max(axis=0)
    
    low, high = target_range
    X_01 = (X - xmin) / (xmax - xmin + 1e-8)
    X_norm = X_01 * (high - low) + low
    
    return X_norm, xmin, xmax


def augment_hf_inputs(X_hf: np.ndarray,
                       X_lf: np.ndarray,
                       Y_lf: np.ndarray,
                       method: str = 'nearest') -> np.ndarray:
    """
    Augment HF inputs with LF predictions.
    
    Creates [X_hf, Y_lf_at_hf] for MF-DNN/KAN architectures.
    
    Args:
        X_hf: HF input locations (N_H, D)
        X_lf: LF input locations (N_L, D)
        Y_lf: LF output values (N_L, 1)
        method: Interpolation method ('nearest' or 'linear')
    
    Returns:
        X_aug: Augmented HF inputs (N_H, D+1)
    """
    Y_lf = np.asarray(Y_lf).flatten()
    
    if method == 'nearest':
        interp = NearestNDInterpolator(X_lf, Y_lf)
    elif method == 'linear':
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(X_lf, Y_lf, fill_value=Y_lf.mean())
    else:
        raise ValueError(f"Unknown method: {method}")
    
    Y_lf_at_hf = interp(X_hf).reshape(-1, 1)
    X_aug = np.hstack([X_hf, Y_lf_at_hf])
    
    return X_aug


def train_test_split_hf(X_hf: np.ndarray,
                         Y_hf: np.ndarray,
                         n_train: int = None,
                         train_ratio: float = None,
                         shuffle: bool = False,
                         seed: int = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split HF data into train/test.
    
    Args:
        X_hf: HF inputs
        Y_hf: HF outputs
        n_train: Number of training points (if specified)
        train_ratio: Training fraction (if n_train not specified)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for shuffling
    
    Returns:
        X_train, Y_train, X_test, Y_test
    """
    n_total = len(X_hf)
    
    if n_train is None:
        if train_ratio is None:
            train_ratio = 0.8
        n_train = int(n_total * train_ratio)
    
    indices = np.arange(n_total)
    
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (X_hf[train_idx], Y_hf[train_idx],
            X_hf[test_idx], Y_hf[test_idx])


def add_gaussian_noise(Y: np.ndarray,
                        noise_level: float = 0.1,
                        relative: bool = True,
                        seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to targets.
    
    Args:
        Y: Target values
        noise_level: Noise level (fraction of std if relative, absolute otherwise)
        relative: If True, noise_level is fraction of Y's std
        seed: Random seed
    
    Returns:
        Y_noisy: Noisy targets
        noise: The noise that was added
    """
    if seed is not None:
        np.random.seed(seed)
    
    if relative:
        sigma = noise_level * np.std(Y)
    else:
        sigma = noise_level
    
    noise = sigma * np.random.randn(*Y.shape)
    Y_noisy = Y + noise
    
    return Y_noisy, noise


def detect_outliers(Y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using z-score threshold.
    
    Args:
        Y: Values to check
        threshold: Z-score threshold (default: 3.0)
    
    Returns:
        outlier_mask: Boolean array (True = outlier)
    """
    Y = np.asarray(Y).flatten()
    z = (Y - np.mean(Y)) / (np.std(Y) + 1e-12)
    return np.abs(z) > threshold


def create_grid(x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 2D grid for visualization.
    
    Returns:
        X_grid, Y_grid: Meshgrid arrays
        XY_flat: Flattened grid points (n_points², 2)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    return X_grid, Y_grid, XY_flat


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing data utilities...")
    
    # Test normalization
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_norm, mean, std = normalize(X)
    X_back = denormalize(X_norm, mean, std)
    assert np.allclose(X, X_back), "Normalization round-trip failed"
    print("✓ Normalization")
    
    # Test augmentation
    X_lf = np.random.rand(100, 2)
    Y_lf = np.sin(X_lf[:, 0])
    X_hf = np.random.rand(10, 2)
    X_aug = augment_hf_inputs(X_hf, X_lf, Y_lf)
    assert X_aug.shape == (10, 3), "Augmentation shape wrong"
    print("✓ Augmentation")
    
    # Test noise
    Y = np.array([1, 2, 3, 4, 5.0])
    Y_noisy, noise = add_gaussian_noise(Y, 0.1, seed=42)
    assert Y_noisy.shape == Y.shape, "Noise shape wrong"
    print("✓ Noise addition")
    
    # Test outlier detection
    Y_with_outlier = np.array([1, 2, 3, 100, 5])
    mask = detect_outliers(Y_with_outlier)
    assert mask[3] == True, "Outlier not detected"
    print("✓ Outlier detection")
    
    print("\n✓ All data utility tests passed!")