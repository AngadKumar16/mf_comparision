"""
Data Utilities

Helper functions for data manipulation, splitting, and augmentation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import NearestNDInterpolator


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


class NormalizingModelWrapper:
    """
    Wraps any MF model with min-max normalization to [-1, 1].

    Callers interact with raw-unit data. The wrapper normalizes before
    fit()/predict() and denormalizes the output, so models never see raw data.

    X stats are fit on combined LF+HF_train.
    Y stats are fit on LF data only (same bounds applied to HF).
    """

    def __init__(self, model):
        self.model = model
        self._X_min = None
        self._X_max = None
        self._Y_min = None
        self._Y_max = None

    # ── helpers ────────────────────────────────────────────────────────────
    @property
    def _yr(self) -> float:
        return float(self._Y_max - self._Y_min) + 1e-8

    def _nx(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * (X - self._X_min) / (self._X_max - self._X_min + 1e-8) - 1.0

    def _ny(self, Y: np.ndarray) -> np.ndarray:
        return 2.0 * (Y - self._Y_min) / self._yr - 1.0

    def _dy(self, Y_n: np.ndarray) -> np.ndarray:
        return (Y_n + 1.0) * self._yr / 2.0 + self._Y_min

    # ── public API ─────────────────────────────────────────────────────────
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray, **kwargs):
        X_all = np.vstack([np.asarray(X_lf), np.asarray(X_hf)])
        self._X_min = X_all.min(axis=0)
        self._X_max = X_all.max(axis=0)
        self._Y_min = float(np.asarray(Y_lf).min())
        self._Y_max = float(np.asarray(Y_lf).max())
        return self.model.fit(
            self._nx(X_lf), self._ny(Y_lf),
            self._nx(X_hf), self._ny(Y_hf),
            **kwargs
        )

    def predict(self, X: np.ndarray, **kwargs):
        y_n, std_n = self.model.predict(self._nx(np.asarray(X)), **kwargs)
        y = self._dy(y_n)
        std = std_n * self._yr / 2.0 if std_n is not None else None
        return y, std

    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        return self._dy(self.model.predict_lf(self._nx(np.asarray(X))))


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
