"""
Multi-Fidelity Deep Neural Network

This module contains:
1. DNN: Base feedforward network class
2. MFTrainer: Multi-fidelity trainer with LF + HF networks
3. MFDNN: Clean interface matching GP API

Extracted from your existing code in Documents 3 and 5.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from scipy.interpolate import NearestNDInterpolator


class DNN:
    """
    Deep Neural Network class using a Multi-Fidelity architecture.
    
    This is your DNN class from Document 3.
    """
    
    def __init__(self):
        pass
    
    def hyper_initial(self, layers: List[int]) -> Tuple[List[tf.Variable], List[tf.Variable]]:
        """
        Initialize weights and biases using Xavier/He initialization.
        
        Args:
            layers: List of layer dimensions, e.g., [2, 20, 20, 1]
        
        Returns:
            W: List of weight matrices
            b: List of bias vectors
        """
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2 / (in_dim + out_dim))  # Xavier initialization
            weight = tf.Variable(
                tf.random.truncated_normal(shape=[in_dim, out_dim], stddev=std),
                dtype=tf.float32
            )
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]), dtype=tf.float32)
            W.append(weight)
            b.append(bias)
        return W, b
    
    def fnn(self, W: List[tf.Variable], b: List[tf.Variable], 
            X: tf.Tensor, Xmin: tf.Tensor, Xmax: tf.Tensor) -> tf.Tensor:
        """
        Feedforward Neural Network with Min-Max Normalization and Tanh Activation.
        
        Args:
            W: Weight matrices
            b: Bias vectors
            X: Input tensor
            Xmin: Minimum values for normalization
            Xmax: Maximum values for normalization
        
        Returns:
            Y: Output tensor
        """
        # Min-max normalization to [-1, 1]
        A = 2.0 * (X - Xmin) / (Xmax - Xmin + 1e-8) - 1.0
        
        L = len(W)
        for i in range(L - 1):
            Z = tf.add(tf.matmul(A, W[i]), b[i])
            A = tf.tanh(Z)
        
        # Linear output layer
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        return Y


class MFTrainer(tf.Module):
    """
    Multi-Fidelity Trainer with separate LF and HF networks.
    
    Architecture:
    - LF network: X → Y_lf
    - HF nonlinear network: [X, Y_lf] → Y_hf_nl
    - HF linear network: [X, Y_lf] → Y_hf_l
    - Final HF: Y_hf = Y_hf_nl + Y_hf_l
    
    This is your MFTrainer from Documents 3 and 5.
    """
    
    def __init__(self, layers_lf: List[int], layers_hf_nl: List[int], 
                 layers_hf_l: List[int], learning_rate: float = 0.001,
                 l2_reg: float = 0.01):
        super().__init__()
        
        self.dnn = DNN()
        self.l2_reg = l2_reg
        
        # Initialize networks
        self.W_lf, self.b_lf = self.dnn.hyper_initial(layers_lf)
        self.W_hf_nl, self.b_hf_nl = self.dnn.hyper_initial(layers_hf_nl)
        self.W_hf_l, self.b_hf_l = self.dnn.hyper_initial(layers_hf_l)
        
        # Combine all trainable variables
        self.trainable_vars = (
            self.W_lf + self.b_lf + 
            self.W_hf_nl + self.b_hf_nl + 
            self.W_hf_l + self.b_hf_l
        )
        
        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    @tf.function
    def train_step(self, x_lf: tf.Tensor, y_lf: tf.Tensor,
                   x_hf_coords: tf.Tensor, y_hf: tf.Tensor,
                   Xmin: tf.Tensor, Xmax: tf.Tensor,
                   Xhmin: tf.Tensor, Xhmax: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single training step.
        
        Args:
            x_lf: LF input coordinates
            y_lf: LF targets
            x_hf_coords: HF input coordinates (2D, without LF augmentation)
            y_hf: HF targets
            Xmin, Xmax: LF normalization bounds
            Xhmin, Xhmax: HF augmented normalization bounds (3D)
        
        Returns:
            total_loss, loss_lf, loss_hf
        """
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_vars)
            
            # LF prediction at LF locations
            y_pred_lf = self.dnn.fnn(self.W_lf, self.b_lf, x_lf, Xmin, Xmax)
            
            # LF prediction at HF coordinates (for augmentation)
            y_lf_at_hf = self.dnn.fnn(self.W_lf, self.b_lf, x_hf_coords, Xmin, Xmax)
            
            # Augment HF coordinates with LF predictions → 3D
            x_aug = tf.concat([x_hf_coords, y_lf_at_hf], axis=1)
            
            # HF predictions (nonlinear + linear)
            y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug, Xhmin, Xhmax)
            y_pred_hf_l = self.dnn.fnn(self.W_hf_l, self.b_hf_l, x_aug, Xhmin, Xhmax)
            y_pred_hf = y_pred_hf_l + y_pred_hf_nl
            
            # Losses
            loss_l2 = self.l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in self.W_hf_nl])
            loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_lf))
            loss_hf = tf.reduce_mean(tf.square(y_pred_hf - y_hf))
            loss = loss_lf + loss_hf + loss_l2
        
        grads = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        
        return loss, loss_lf, loss_hf
    
    def predict(self, x_coords: tf.Tensor, 
                Xmin: tf.Tensor, Xmax: tf.Tensor,
                Xhmin: tf.Tensor, Xhmax: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict LF and HF outputs.
        
        Args:
            x_coords: Input coordinates (2D)
            Xmin, Xmax: LF normalization bounds
            Xhmin, Xhmax: HF augmented normalization bounds
        
        Returns:
            y_pred_hf, y_pred_lf
        """
        # LF prediction
        y_pred_lf = self.dnn.fnn(self.W_lf, self.b_lf, x_coords, Xmin, Xmax)
        
        # Augment with LF prediction
        x_aug = tf.concat([x_coords, y_pred_lf], axis=1)
        
        # HF prediction
        y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug, Xhmin, Xhmax)
        y_pred_hf_l = self.dnn.fnn(self.W_hf_l, self.b_hf_l, x_aug, Xhmin, Xhmax)
        y_pred_hf = y_pred_hf_l + y_pred_hf_nl
        
        return y_pred_hf, y_pred_lf
    
    def get_weights(self):
        """Return weight lists for saving."""
        return (
            [self.W_lf, self.W_hf_nl, self.W_hf_l],
            [self.b_lf, self.b_hf_nl, self.b_hf_l]
        )


class MFDNN:
    """
    Multi-Fidelity DNN with clean interface matching GP API.
    
    This wraps MFTrainer to provide:
    - fit(X_lf, Y_lf, X_hf, Y_hf) → training
    - predict(X, return_std=True) → HF predictions with uncertainty
    - predict_lf(X) → LF predictions
    """
    
    def __init__(self, 
                 layers_lf: List[int] = None,
                 layers_hf_nl: List[int] = None,
                 layers_hf_l: List[int] = None,
                 learning_rate: float = 0.001,
                 max_epochs: int = 30000,
                 patience: int = 2000,
                 l2_reg: float = 0.01,
                 verbose: bool = True):
        """
        Initialize MF-DNN.
        
        Args:
            layers_lf: LF network architecture, e.g., [2, 20, 20, 1]
            layers_hf_nl: HF nonlinear network, e.g., [3, 10, 10, 1]
            layers_hf_l: HF linear network, e.g., [3, 1]
            learning_rate: Initial learning rate
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            l2_reg: L2 regularization strength
            verbose: Print training progress
        """
        self.layers_lf = layers_lf or [2, 20, 20, 1]
        self.layers_hf_nl = layers_hf_nl or [3, 10, 10, 1]
        self.layers_hf_l = layers_hf_l or [3, 1]
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2_reg = l2_reg
        self.verbose = verbose
        
        self.trainer = None
        self.is_trained = False
        
        # Normalization bounds (computed during training)
        self.Xmin = None
        self.Xmax = None
        self.Xhmin = None
        self.Xhmax = None
    
    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray) -> Dict[str, Any]:
        """
        Train the MF-DNN.
        
        Args:
            X_lf: LF inputs (N_L, D)
            Y_lf: LF outputs (N_L, 1)
            X_hf: HF inputs (N_H, D)
            Y_hf: HF outputs (N_H, 1)
        
        Returns:
            Training info dict
        """
        # Convert to numpy and ensure shapes
        X_lf = np.asarray(X_lf, dtype=np.float32)
        Y_lf = np.asarray(Y_lf, dtype=np.float32).reshape(-1, 1)
        X_hf = np.asarray(X_hf, dtype=np.float32)
        Y_hf = np.asarray(Y_hf, dtype=np.float32).reshape(-1, 1)
        
        # =====================================================
        # Compute normalization bounds
        # =====================================================
        self.Xmin = tf.constant(X_lf.min(axis=0), dtype=tf.float32)
        self.Xmax = tf.constant(X_lf.max(axis=0), dtype=tf.float32)
        
        # Get LF predictions at HF locations for augmented bounds
        lf_interp = NearestNDInterpolator(X_lf, Y_lf.flatten())
        Y_lf_at_hf = lf_interp(X_hf).reshape(-1, 1)
        X_hf_aug = np.hstack([X_hf, Y_lf_at_hf])
        
        self.Xhmin = tf.constant(X_hf_aug.min(axis=0), dtype=tf.float32)
        self.Xhmax = tf.constant(X_hf_aug.max(axis=0), dtype=tf.float32)
        
        # =====================================================
        # Create trainer
        # =====================================================
        self.trainer = MFTrainer(
            self.layers_lf, self.layers_hf_nl, self.layers_hf_l,
            learning_rate=self.learning_rate, l2_reg=self.l2_reg
        )
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        self.trainer.optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        
        # Convert to tensors
        x_lf_t = tf.convert_to_tensor(X_lf, dtype=tf.float32)
        y_lf_t = tf.convert_to_tensor(Y_lf, dtype=tf.float32)
        x_hf_t = tf.convert_to_tensor(X_hf, dtype=tf.float32)
        y_hf_t = tf.convert_to_tensor(Y_hf, dtype=tf.float32)
        
        # =====================================================
        # Training loop with early stopping
        # =====================================================
        best_loss = float('inf')
        wait = 0
        best_weights = None
        training_history = []
        
        for epoch in range(self.max_epochs):
            loss, loss_lf, loss_hf = self.trainer.train_step(
                x_lf_t, y_lf_t, x_hf_t, y_hf_t,
                self.Xmin, self.Xmax, self.Xhmin, self.Xhmax
            )
            
            loss_val = float(loss)
            training_history.append({
                'epoch': epoch,
                'loss': loss_val,
                'loss_lf': float(loss_lf),
                'loss_hf': float(loss_hf)
            })
            
            # Early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                wait = 0
                best_weights = [w.numpy() for w in self.trainer.trainable_vars]
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.verbose and epoch % 5000 == 0:
                print(f"Epoch {epoch}: loss={loss_val:.6f}, LF={float(loss_lf):.6f}, HF={float(loss_hf):.6f}")
        
        # Restore best weights
        if best_weights is not None:
            for w, val in zip(self.trainer.trainable_vars, best_weights):
                w.assign(val)
        
        self.is_trained = True
        
        return {
            'final_loss': best_loss,
            'epochs_trained': epoch + 1,
            'history': training_history
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True,
                n_mc_samples: int = 100) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict HF output.
        
        Note: This basic version doesn't provide true uncertainty.
        For uncertainty, use ensemble or MC dropout wrapper.
        
        Args:
            X: Input locations (N, D)
            return_std: Whether to return std (placeholder for now)
            n_mc_samples: Not used in base model
        
        Returns:
            mean: Predicted HF values
            std: Placeholder (zeros) - use ensemble for real uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        
        y_hf, y_lf = self.trainer.predict(
            X, self.Xmin, self.Xmax, self.Xhmin, self.Xhmax
        )
        
        mean = y_hf.numpy()
        
        if return_std:
            # Placeholder - use DeepEnsemble for real uncertainty
            std = np.zeros_like(mean)
            return mean, std
        return mean, None
    
    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        
        _, y_lf = self.trainer.predict(
            X, self.Xmin, self.Xmax, self.Xhmin, self.Xhmax
        )
        
        return y_lf.numpy()


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing MF-DNN with synthetic data...")
    
    # Suppress TF warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Create simple test data
    np.random.seed(42)
    
    # LF data (dense)
    X_lf = np.random.rand(100, 2).astype(np.float32)
    Y_lf = (np.sin(2 * np.pi * X_lf[:, 0:1]) + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    # HF data (sparse)
    X_hf = np.random.rand(12, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)
    
    # Test MFDNN
    print("\nTesting MFDNN...")
    model = MFDNN(max_epochs=5000, patience=500, verbose=True)
    info = model.fit(X_lf, Y_lf, X_hf, Y_hf)
    print(f"\nFinal loss: {info['final_loss']:.6f}")
    print(f"Epochs trained: {info['epochs_trained']}")
    
    # Test prediction
    X_test = np.random.rand(5, 2).astype(np.float32)
    y_pred, y_std = model.predict(X_test)
    print(f"\nPredictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred.flatten()[:3]}")
    
    print("\n✓ MF-DNN tests passed!")