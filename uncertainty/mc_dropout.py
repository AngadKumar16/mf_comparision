"""
Monte Carlo Dropout for Uncertainty Quantification

Enables dropout at test time for Bayesian approximation.
Multiple forward passes with dropout give uncertainty estimates.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning" (2016)
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any


class MCDropoutWrapper:
    """
    MC Dropout wrapper for any TensorFlow model.
    
    Usage:
        model = MFDNN(...)  # Must have dropout layers
        model.fit(X_lf, Y_lf, X_hf, Y_hf)
        
        mc_model = MCDropoutWrapper(model, dropout_rate=0.1)
        mean, std = mc_model.predict(X_test, n_samples=100)
    """
    
    def __init__(self, model, dropout_rate: float = 0.1):
        """
        Args:
            model: Trained model with predict method
            dropout_rate: Dropout probability (for reference)
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.name = f"MC-Dropout({model.name if hasattr(model, 'name') else 'model'})"
    
    def predict(self, X: np.ndarray, n_samples: int = 100,
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        MC Dropout prediction.
        
        Args:
            X: Input locations
            n_samples: Number of forward passes with dropout
            return_std: Whether to return uncertainty
        
        Returns:
            mean: Average prediction across samples
            std: Std across samples (epistemic uncertainty)
        """
        predictions = []
        
        for _ in range(n_samples):
            # Forward pass with dropout enabled (training=True)
            if hasattr(self.model, 'predict_with_dropout'):
                y_pred = self.model.predict_with_dropout(X, training=True)
            else:
                # Fallback: use regular predict
                y_pred, _ = self.model.predict(X, return_std=False)
            
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        
        if return_std:
            std = np.std(predictions, axis=0)
            return mean, std
        return mean, None
    
    def predict_with_ci(self, X: np.ndarray, n_samples: int = 100,
                        ci_level: float = 0.95
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prediction with confidence intervals.
        
        Returns:
            mean: Mean prediction
            lower: Lower CI bound
            upper: Upper CI bound
        """
        predictions = []
        
        for _ in range(n_samples):
            if hasattr(self.model, 'predict_with_dropout'):
                y_pred = self.model.predict_with_dropout(X, training=True)
            else:
                y_pred, _ = self.model.predict(X, return_std=False)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis=0)
        alpha = (1 - ci_level) / 2
        lower = np.percentile(predictions, alpha * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return mean, lower, upper


class MCDropoutDNN(tf.Module):
    """
    Multi-Fidelity DNN with explicit MC Dropout support.
    
    This is a modified version of MFDNN that ensures dropout
    is properly enabled during inference for MC sampling.
    """
    
    def __init__(self, layers_lf, layers_hf_nl, layers_hf_l,
                 dropout_rate: float = 0.1, learning_rate: float = 0.001):
        super().__init__(name="MCDropoutDNN")

        self.dropout_rate = dropout_rate
        self.layers_lf = self._build_layers(layers_lf, dropout_rate)
        self.layers_hf_nl = self._build_layers(layers_hf_nl, dropout_rate)
        self.layers_hf_l = self._build_linear_layers(layers_hf_l)

        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.is_trained = False
        self.model_name = "MC-Dropout-DNN"
        
        # Normalization stats
        self.Xmin = None
        self.Xmax = None
        self.Xhmin = None
        self.Xhmax = None
    
    def _build_layers(self, layer_sizes, dropout_rate):
        """Build layers with dropout after each hidden layer."""
        layers = []
        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            
            layers.append(tf.keras.layers.Dense(
                layer_sizes[i + 1],
                activation=None if is_output else 'tanh',
                kernel_initializer='glorot_uniform'
            ))
            
            # Add dropout after hidden layers (not output)
            if not is_output:
                layers.append(tf.keras.layers.Dropout(dropout_rate))
        
        return layers
    
    def _build_linear_layers(self, layer_sizes):
        """Build linear layers (no dropout, no activation)."""
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(tf.keras.layers.Dense(
                layer_sizes[i + 1],
                activation=None
            ))
        return layers
    
    def _forward(self, x, layers, training=False):
        """Forward pass through layer stack."""
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    
    def _normalize(self, x, xmin, xmax):
        """Min-max normalize to [-1, 1]."""
        return 2.0 * (x - xmin) / (xmax - xmin + 1e-8) - 1.0
    
    @tf.function
    def train_step(self, x_lf, y_lf, x_hf, y_hf):
        """Training step with dropout enabled."""
        with tf.GradientTape() as tape:
            # LF prediction
            x_lf_n = self._normalize(x_lf, self.Xmin, self.Xmax)
            y_lf_pred = self._forward(x_lf_n, self.layers_lf, training=True)
            
            # LF at HF locations
            x_hf_n = self._normalize(x_hf, self.Xmin, self.Xmax)
            y_lf_at_hf = self._forward(x_hf_n, self.layers_lf, training=True)
            
            # Augment and predict HF
            x_aug = tf.concat([x_hf, y_lf_at_hf], axis=1)
            x_aug_n = self._normalize(x_aug, self.Xhmin, self.Xhmax)
            
            y_hf_nl = self._forward(x_aug_n, self.layers_hf_nl, training=True)
            y_hf_l = self._forward(x_aug_n, self.layers_hf_l, training=True)
            y_hf_pred = y_hf_nl + y_hf_l
            
            # Losses
            loss_lf = tf.reduce_mean(tf.square(y_lf_pred - y_lf))
            loss_hf = tf.reduce_mean(tf.square(y_hf_pred - y_hf))
            loss = loss_lf + loss_hf
        
        # Collect trainable variables
        variables = []
        for layer in self.layers_lf + self.layers_hf_nl + self.layers_hf_l:
            if hasattr(layer, 'trainable_variables'):
                variables.extend(layer.trainable_variables)
        
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        
        return loss, loss_lf, loss_hf
    
    def predict_with_dropout(self, X, training=True):
        """
        Forward pass with dropout enabled (for MC sampling).
        
        Args:
            X: Input locations
            training: If True, dropout is active
        
        Returns:
            y_hf: HF predictions
        """
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        
        # LF prediction
        x_n = self._normalize(X, self.Xmin, self.Xmax)
        y_lf = self._forward(x_n, self.layers_lf, training=training)
        
        # Augment
        x_aug = tf.concat([X, y_lf], axis=1)
        x_aug_n = self._normalize(x_aug, self.Xhmin, self.Xhmax)
        
        # HF prediction
        y_hf_nl = self._forward(x_aug_n, self.layers_hf_nl, training=training)
        y_hf_l = self._forward(x_aug_n, self.layers_hf_l, training=training)
        
        return (y_hf_nl + y_hf_l).numpy()
    
    def fit(self, X_lf, Y_lf, X_hf, Y_hf,
            max_epochs=30000, patience=2000, verbose=True):
        """Train the model."""
        X_lf = np.asarray(X_lf, dtype=np.float32)
        Y_lf = np.asarray(Y_lf, dtype=np.float32).reshape(-1, 1)
        X_hf = np.asarray(X_hf, dtype=np.float32)
        Y_hf = np.asarray(Y_hf, dtype=np.float32).reshape(-1, 1)
        
        # Compute normalization bounds
        self.Xmin = tf.constant(X_lf.min(axis=0), dtype=tf.float32)
        self.Xmax = tf.constant(X_lf.max(axis=0), dtype=tf.float32)
        
        # Get LF predictions at HF for augmented bounds
        from scipy.interpolate import NearestNDInterpolator
        lf_interp = NearestNDInterpolator(X_lf, Y_lf.flatten())
        Y_lf_at_hf = lf_interp(X_hf).reshape(-1, 1)
        X_hf_aug = np.hstack([X_hf, Y_lf_at_hf])
        
        self.Xhmin = tf.constant(X_hf_aug.min(axis=0), dtype=tf.float32)
        self.Xhmax = tf.constant(X_hf_aug.max(axis=0), dtype=tf.float32)
        
        # Convert to tensors
        x_lf_t = tf.constant(X_lf, dtype=tf.float32)
        y_lf_t = tf.constant(Y_lf, dtype=tf.float32)
        x_hf_t = tf.constant(X_hf, dtype=tf.float32)
        y_hf_t = tf.constant(Y_hf, dtype=tf.float32)
        
        # Collect trainable variables from all layers
        all_layers = self.layers_lf + self.layers_hf_nl + self.layers_hf_l
        trainable_vars = [v for layer in all_layers
                          if hasattr(layer, 'trainable_variables')
                          for v in layer.trainable_variables]

        # Training loop
        best_loss = float('inf')
        wait = 0
        best_weights = None

        for epoch in range(max_epochs):
            loss, loss_lf, loss_hf = self.train_step(
                x_lf_t, y_lf_t, x_hf_t, y_hf_t
            )

            loss_val = float(loss)

            if loss_val < best_loss:
                best_loss = loss_val
                wait = 0
                best_weights = [v.numpy() for v in trainable_vars]
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 5000 == 0:
                print(f"Epoch {epoch}: loss={loss_val:.6f}")

        # Restore best weights
        if best_weights is not None:
            for v, val in zip(trainable_vars, best_weights):
                v.assign(val)

        self.is_trained = True
        return {'final_loss': best_loss, 'epochs': epoch + 1}
    
    def predict(self, X, return_std=True, n_samples=100):
        """Predict with MC Dropout uncertainty."""
        if return_std:
            predictions = []
            for _ in range(n_samples):
                y_pred = self.predict_with_dropout(X, training=True)
                predictions.append(y_pred)
            
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            return mean, std
        else:
            return self.predict_with_dropout(X, training=False), None
    
    def predict_lf(self, X):
        """Predict LF output."""
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        x_n = self._normalize(X, self.Xmin, self.Xmax)
        y_lf = self._forward(x_n, self.layers_lf, training=False)
        return y_lf.numpy()


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing MC Dropout...")
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    np.random.seed(42)
    
    # Synthetic data
    X_lf = np.random.rand(50, 2).astype(np.float32)
    Y_lf = np.sin(2 * np.pi * X_lf[:, 0:1]).astype(np.float32)
    
    X_hf = np.random.rand(10, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)
    
    # Train MC Dropout DNN
    model = MCDropoutDNN(
        layers_lf=[2, 20, 1],
        layers_hf_nl=[3, 20, 1],
        layers_hf_l=[3, 1],
        dropout_rate=0.1
    )
    
    info = model.fit(X_lf, Y_lf, X_hf, Y_hf, max_epochs=3000, patience=300, verbose=True)
    print(f"\nTraining complete: {info}")
    
    # Predict with uncertainty
    X_test = np.random.rand(5, 2).astype(np.float32)
    mean, std = model.predict(X_test, return_std=True, n_samples=50)
    
    print(f"\nPredictions: {mean.flatten()}")
    print(f"Uncertainties: {std.flatten()}")
    
    print("\n✓ MC Dropout test passed!")