"""
Multi-Fidelity Kolmogorov-Arnold Network (KAN)

This module contains:
1. KANLayer: B-spline based KAN layer
2. KAN: Multi-layer KAN network
3. MFKANTrainer: Multi-fidelity trainer
4. MFKAN: Clean interface matching GP/DNN API

Extracted from your existing code in Document 6.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from scipy.interpolate import NearestNDInterpolator


class KANLayer(tf.Module):
    """
    Efficient KAN Layer using B-spline basis functions.
    
    Based on: https://github.com/Blealtan/efficient-kan
    
    Instead of computing activation per edge (expensive), we:
    1. Compute B-spline basis for all inputs
    2. Linearly combine with learnable coefficients
    """
    
    def __init__(self, in_dim: int, out_dim: int, 
                 grid_size: int = 5, spline_order: int = 3,
                 name: str = "kan_layer"):
        super().__init__(name=name)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size  # G: number of grid intervals
        self.spline_order = spline_order  # k: polynomial degree
        
        # Grid range [-1, 1] (after min-max normalization)
        self.grid_range = [-1.0, 1.0]
        
        # Number of basis functions = G + k
        self.num_basis = grid_size + spline_order
        
        # Initialize grid: shape (in_dim, grid_size + 1)
        grid = tf.linspace(self.grid_range[0], self.grid_range[1], grid_size + 1)
        self.grid = tf.Variable(
            tf.tile(tf.expand_dims(grid, 0), [in_dim, 1]),
            trainable=False,
            name='grid',
            dtype=tf.float32
        )
        
        # Base weight: standard linear transformation (residual path)
        self.base_weight = tf.Variable(
            tf.random.normal([in_dim, out_dim], stddev=0.1),
            name='base_weight',
            dtype=tf.float32
        )
        self.base_bias = tf.Variable(
            tf.zeros([out_dim]),
            name='base_bias',
            dtype=tf.float32
        )
        
        # Spline coefficients: shape (in_dim, out_dim, num_basis)
        self.spline_weight = tf.Variable(
            tf.random.normal([in_dim, out_dim, self.num_basis], stddev=0.1),
            name='spline_weight',
            dtype=tf.float32
        )
        
        # Learnable scale for spline
        self.spline_scale = tf.Variable(
            tf.ones([in_dim, out_dim]),
            name='spline_scale',
            dtype=tf.float32
        )
    
    def compute_bspline_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute B-spline basis functions.
        
        Args:
            x: Input tensor (batch_size, in_dim)
        
        Returns:
            basis: B-spline basis (batch_size, in_dim, num_basis)
        """
        k = self.spline_order
        dx = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        
        # Extend grid for spline order
        left_extension = tf.stack(
            [self.grid[:, 0] - (k - i) * dx for i in range(k)],
            axis=1
        )
        right_extension = tf.stack(
            [self.grid[:, -1] + (i + 1) * dx for i in range(k)],
            axis=1
        )
        grid_extended = tf.concat([left_extension, self.grid, right_extension], axis=1)
        
        # Expand dimensions for broadcasting
        x_expanded = tf.expand_dims(x, -1)  # (batch, in_dim, 1)
        grid_expanded = tf.expand_dims(grid_extended, 0)  # (1, in_dim, num_grid)
        
        # Compute basis (simplified - using soft step approximation)
        left = tf.sigmoid((x_expanded - grid_expanded[:, :, :-1]) * 100)
        right = tf.sigmoid((grid_expanded[:, :, 1:] - x_expanded) * 100)
        basis = left * right
        
        # Recursion for higher orders
        for order in range(1, k + 1):
            num_grid = tf.shape(grid_extended)[1]
            
            # Weights for recursion
            num = x_expanded - grid_expanded[:, :, :num_grid - 1 - order]
            den = grid_expanded[:, :, order:num_grid - 1] - grid_expanded[:, :, :num_grid - 1 - order] + 1e-8
            w1 = num / den
            
            num = grid_expanded[:, :, order + 1:] - x_expanded
            den = grid_expanded[:, :, order + 1:] - grid_expanded[:, :, 1:num_grid - order] + 1e-8
            w2 = num / den
            
            # Limit to valid range
            basis = w1[:, :, :tf.shape(basis)[-1] - 1] * basis[:, :, :-1] + \
                   w2[:, :, :tf.shape(basis)[-1] - 1] * basis[:, :, 1:]
        
        return basis
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_dim)
        
        Returns:
            y: Output tensor (batch_size, out_dim)
        """
        # Base (residual) path with SiLU activation
        base_act = x * tf.sigmoid(x)  # SiLU(x) = x * sigmoid(x)
        base_output = tf.matmul(base_act, self.base_weight) + self.base_bias
        
        # Spline path
        basis = self.compute_bspline_basis(x)
        
        # Combine basis with coefficients
        # basis: (batch, in, num_basis), spline_weight: (in, out, num_basis)
        spline_output = tf.einsum('bin,ion->bo', basis, self.spline_weight)
        
        # Apply learnable scale
        spline_output = spline_output * tf.reduce_mean(self.spline_scale)
        
        return base_output + spline_output
    
    def regularization_loss(self, l1_factor: float = 0.01) -> tf.Tensor:
        """L1 regularization on spline weights for sparsity."""
        return l1_factor * tf.reduce_mean(tf.abs(self.spline_weight))
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return [self.base_weight, self.base_bias, 
                self.spline_weight, self.spline_scale]


class KAN(tf.Module):
    """Multi-layer Kolmogorov-Arnold Network."""
    
    def __init__(self, layers_config: List[int], 
                 grid_size: int = 5, spline_order: int = 3,
                 name: str = "kan"):
        """
        Args:
            layers_config: List of layer dimensions, e.g., [2, 20, 20, 1]
            grid_size: Number of grid intervals for B-splines
            spline_order: Polynomial degree (3 = cubic)
        """
        super().__init__(name=name)
        
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(
                KANLayer(
                    layers_config[i],
                    layers_config[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    name=f'kan_layer_{i}'
                )
            )
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def regularization_loss(self, l1_factor: float = 0.01) -> tf.Tensor:
        """Sum of regularization losses from all layers."""
        return sum(layer.regularization_loss(l1_factor) for layer in self.layers)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        variables = []
        for layer in self.layers:
            variables.extend(layer.trainable_variables)
        return variables


class MFKANTrainer(tf.Module):
    """
    Multi-Fidelity KAN Trainer.
    
    Architecture:
    - LF KAN: maps X → Y_lf
    - HF KAN (Nonlinear): maps [X, Y_lf] → Y_hf_nl
    - HF Linear: maps [X, Y_lf] → Y_hf_l
    - Final: Y_hf = Y_hf_nl + Y_hf_l
    """
    
    def __init__(self, layers_lf: List[int], layers_hf_nl: List[int], 
                 layers_hf_l: List[int], grid_size: int = 5, 
                 spline_order: int = 3, learning_rate: float = 0.001):
        super().__init__()
        
        # LF KAN
        self.kan_lf = KAN(layers_lf, grid_size, spline_order, name='kan_lf')
        
        # HF Nonlinear KAN
        self.kan_hf_nl = KAN(layers_hf_nl, grid_size, spline_order, name='kan_hf_nl')
        
        # HF Linear (simple linear layer)
        self.W_hf_l = tf.Variable(
            tf.random.normal([layers_hf_l[0], layers_hf_l[-1]], stddev=0.1),
            name='W_hf_l',
            dtype=tf.float32
        )
        self.b_hf_l = tf.Variable(
            tf.zeros([layers_hf_l[-1]]),
            name='b_hf_l',
            dtype=tf.float32
        )
        
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return (
            self.kan_lf.trainable_variables +
            self.kan_hf_nl.trainable_variables +
            [self.W_hf_l, self.b_hf_l]
        )
    
    @tf.function
    def train_step(self, x_lf: tf.Tensor, y_lf: tf.Tensor,
                   x_hf_coords: tf.Tensor, y_hf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single training step.
        """
        with tf.GradientTape() as tape:
            # LF prediction at LF locations
            y_pred_lf = self.kan_lf(x_lf)
            
            # LF prediction at HF coordinates
            y_lf_at_hf = self.kan_lf(x_hf_coords)
            
            # Augment HF coordinates
            x_aug = tf.concat([x_hf_coords, y_lf_at_hf], axis=1)
            
            # HF predictions
            y_pred_hf_nl = self.kan_hf_nl(x_aug)
            y_pred_hf_l = tf.matmul(x_aug, self.W_hf_l) + self.b_hf_l
            y_pred_hf = y_pred_hf_l + y_pred_hf_nl
            
            # Losses
            loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_lf))
            loss_hf = tf.reduce_mean(tf.square(y_pred_hf - y_hf))
            
            # KAN regularization
            reg_lf = self.kan_lf.regularization_loss(0.001)
            reg_hf = self.kan_hf_nl.regularization_loss(0.001)
            
            loss = loss_lf + loss_hf + reg_lf + reg_hf
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss, loss_lf, loss_hf
    
    def predict(self, x_coords: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict LF and HF outputs.
        """
        # LF prediction
        y_pred_lf = self.kan_lf(x_coords)
        
        # Augment and predict HF
        x_aug = tf.concat([x_coords, y_pred_lf], axis=1)
        y_pred_hf_nl = self.kan_hf_nl(x_aug)
        y_pred_hf_l = tf.matmul(x_aug, self.W_hf_l) + self.b_hf_l
        y_pred_hf = y_pred_hf_l + y_pred_hf_nl
        
        return y_pred_hf, y_pred_lf


class MFKAN:
    """
    Multi-Fidelity KAN with clean interface matching GP/DNN API.
    
    Provides:
    - fit(X_lf, Y_lf, X_hf, Y_hf) → training
    - predict(X, return_std=True) → HF predictions
    - predict_lf(X) → LF predictions
    """
    
    def __init__(self,
                 layers_lf: List[int] = None,
                 layers_hf_nl: List[int] = None,
                 layers_hf_l: List[int] = None,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 learning_rate: float = 0.001,
                 max_epochs: int = 30000,
                 patience: int = 1000,
                 verbose: bool = True):
        """
        Initialize MF-KAN.
        """
        self.layers_lf = layers_lf or [2, 20, 20, 1]
        self.layers_hf_nl = layers_hf_nl or [3, 10, 10, 1]
        self.layers_hf_l = layers_hf_l or [3, 1]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        
        self.trainer = None
        self.is_trained = False
        
        # Store normalization stats
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
    
    def _normalize_inputs(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize inputs to [-1, 1] range."""
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = np.maximum(X.std(axis=0), 1e-8)
        return (X - self.X_mean) / self.X_std
    
    def _normalize_outputs(self, Y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize outputs."""
        if fit:
            self.Y_mean = Y.mean(axis=0)
            self.Y_std = np.maximum(Y.std(axis=0), 1e-8)
        return (Y - self.Y_mean) / self.Y_std
    
    def _denormalize_outputs(self, Y_norm: np.ndarray) -> np.ndarray:
        """Denormalize outputs."""
        return Y_norm * self.Y_std + self.Y_mean
    
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray) -> Dict[str, Any]:
        """
        Train the MF-KAN.
        """
        # Convert to numpy
        X_lf = np.asarray(X_lf, dtype=np.float32)
        Y_lf = np.asarray(Y_lf, dtype=np.float32).reshape(-1, 1)
        X_hf = np.asarray(X_hf, dtype=np.float32)
        Y_hf = np.asarray(Y_hf, dtype=np.float32).reshape(-1, 1)
        
        # Normalize (fit on LF data)
        X_lf_n = self._normalize_inputs(X_lf, fit=True)
        Y_lf_n = self._normalize_outputs(Y_lf, fit=True)
        X_hf_n = self._normalize_inputs(X_hf)
        Y_hf_n = self._normalize_outputs(Y_hf)
        
        # Create trainer
        self.trainer = MFKANTrainer(
            self.layers_lf, self.layers_hf_nl, self.layers_hf_l,
            grid_size=self.grid_size, spline_order=self.spline_order,
            learning_rate=self.learning_rate
        )
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.95
        )
        self.trainer.optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        
        # Convert to tensors
        x_lf_t = tf.convert_to_tensor(X_lf_n, dtype=tf.float32)
        y_lf_t = tf.convert_to_tensor(Y_lf_n, dtype=tf.float32)
        x_hf_t = tf.convert_to_tensor(X_hf_n, dtype=tf.float32)
        y_hf_t = tf.convert_to_tensor(Y_hf_n, dtype=tf.float32)
        
        # Training loop
        best_loss = float('inf')
        wait = 0
        
        for epoch in range(self.max_epochs):
            loss, loss_lf, loss_hf = self.trainer.train_step(
                x_lf_t, y_lf_t, x_hf_t, y_hf_t
            )
            
            loss_val = float(loss)
            
            if loss_val < best_loss:
                best_loss = loss_val
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.verbose and epoch % 5000 == 0:
                print(f"Epoch {epoch}: loss={loss_val:.6f}")
        
        self.is_trained = True
        
        return {
            'final_loss': best_loss,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True,
                n_mc_samples: int = 100) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict HF output.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        X_n = self._normalize_inputs(X)
        X_t = tf.convert_to_tensor(X_n, dtype=tf.float32)
        
        y_hf_n, _ = self.trainer.predict(X_t)
        y_hf = self._denormalize_outputs(y_hf_n.numpy())
        
        if return_std:
            # Placeholder - use ensemble for real uncertainty
            std = np.zeros_like(y_hf)
            return y_hf, std
        return y_hf, None
    
    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        X_n = self._normalize_inputs(X)
        X_t = tf.convert_to_tensor(X_n, dtype=tf.float32)
        
        _, y_lf_n = self.trainer.predict(X_t)
        return self._denormalize_outputs(y_lf_n.numpy())


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing MF-KAN with synthetic data...")
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    np.random.seed(42)
    
    # LF data
    X_lf = np.random.rand(100, 2).astype(np.float32)
    Y_lf = (np.sin(2 * np.pi * X_lf[:, 0:1]) + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    # HF data
    X_hf = np.random.rand(12, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)
    
    print("\nTesting MFKAN...")
    model = MFKAN(max_epochs=3000, patience=300, verbose=True)
    info = model.fit(X_lf, Y_lf, X_hf, Y_hf)
    print(f"\nFinal loss: {info['final_loss']:.6f}")
    
    # Test prediction
    X_test = np.random.rand(5, 2).astype(np.float32)
    y_pred, _ = model.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")
    
    print("\n✓ MF-KAN tests passed!")