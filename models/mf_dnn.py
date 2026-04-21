"""
Multi-Fidelity Deep Neural Network

This module contains:
1. DNN: Base feedforward network class
2. MFTrainer: Multi-fidelity trainer with LF + HF networks
3. MFDNN: Clean interface matching GP API

Normalization is handled externally by NormalizingModelWrapper in utils/data_utils.py.
Models receive pre-normalized data in [-1, 1] and return normalized outputs.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class DNN:
    """
    Deep Neural Network class using a Multi-Fidelity architecture.
    """

    def __init__(self):
        pass

    def hyper_initial(self, layers: List[int]) -> Tuple[List[tf.Variable], List[tf.Variable]]:
        """
        Initialize weights and biases using Xavier initialization.

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
            std = np.sqrt(2 / (in_dim + out_dim))
            weight = tf.Variable(
                tf.random.truncated_normal(shape=[in_dim, out_dim], stddev=std),
                dtype=tf.float32
            )
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]), dtype=tf.float32)
            W.append(weight)
            b.append(bias)
        return W, b

    def fnn(self, W: List[tf.Variable], b: List[tf.Variable],
            X: tf.Tensor) -> tf.Tensor:
        """
        Feedforward Neural Network with Tanh activation.

        Expects pre-normalized inputs in [-1, 1] (handled by NormalizingModelWrapper).

        Args:
            W: Weight matrices
            b: Bias vectors
            X: Input tensor (pre-normalized)

        Returns:
            Y: Output tensor (normalized space)
        """
        A = X
        L = len(W)
        for i in range(L - 1):
            Z = tf.add(tf.matmul(A, W[i]), b[i])
            A = tf.tanh(Z)
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

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.lf_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, x_lf: tf.Tensor, y_lf: tf.Tensor,
                   x_hf_coords: tf.Tensor, y_hf: tf.Tensor
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single training step. All inputs pre-normalized to [-1, 1].

        Args:
            x_lf: LF input coordinates (pre-normalized)
            y_lf: LF targets (pre-normalized)
            x_hf_coords: HF input coordinates (pre-normalized, 2D)
            y_hf: HF targets (pre-normalized)

        Returns:
            total_loss, loss_lf, loss_hf
        """
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_vars)

            # LF prediction at LF locations
            y_pred_lf = self.dnn.fnn(self.W_lf, self.b_lf, x_lf)

            # LF prediction at HF coordinates (for augmentation)
            y_lf_at_hf = self.dnn.fnn(self.W_lf, self.b_lf, x_hf_coords)

            # Augment HF coordinates with LF predictions → 3D
            x_aug = tf.concat([x_hf_coords, y_lf_at_hf], axis=1)

            # HF predictions (nonlinear + linear)
            y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug)
            y_pred_hf_l = self.dnn.fnn(self.W_hf_l, self.b_hf_l, x_aug)
            y_pred_hf = y_pred_hf_l + y_pred_hf_nl

            # Losses
            loss_l2 = self.l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in self.W_hf_nl])
            loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_lf))
            loss_hf = tf.reduce_mean(tf.square(y_pred_hf - y_hf))
            loss = loss_lf + loss_hf + loss_l2

        grads = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))

        return loss, loss_lf, loss_hf

    def predict(self, x_coords: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict LF and HF outputs. Inputs pre-normalized; outputs in normalized space.

        Args:
            x_coords: Input coordinates (2D, pre-normalized)

        Returns:
            y_pred_hf, y_pred_lf
        """
        y_pred_lf = self.dnn.fnn(self.W_lf, self.b_lf, x_coords)
        x_aug = tf.concat([x_coords, y_pred_lf], axis=1)
        y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug)
        y_pred_hf_l = self.dnn.fnn(self.W_hf_l, self.b_hf_l, x_aug)
        y_pred_hf = y_pred_hf_l + y_pred_hf_nl
        return y_pred_hf, y_pred_lf

    def pretrain_step_lf(self, x_lf: tf.Tensor, y_lf: tf.Tensor) -> tf.Tensor:
        """Single LF-only training step (Phase 1 pretraining)."""
        lf_vars = self.W_lf + self.b_lf
        with tf.GradientTape() as tape:
            tape.watch(lf_vars)
            y_pred_lf = self.dnn.fnn(self.W_lf, self.b_lf, x_lf)
            loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_lf))
        grads = tape.gradient(loss_lf, lf_vars)
        self.lf_optimizer.apply_gradients(zip(grads, lf_vars))
        return loss_lf

    def get_weights(self):
        """Return weight lists for saving."""
        return (
            [self.W_lf, self.W_hf_nl, self.W_hf_l],
            [self.b_lf, self.b_hf_nl, self.b_hf_l]
        )


class MFDNN:
    """
    Multi-Fidelity DNN with clean interface matching GP API.

    Expects pre-normalized data (handled by NormalizingModelWrapper).
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
                 lf_pretrain_patience: int = 500,
                 verbose: bool = True):
        self.layers_lf = layers_lf or [2, 20, 20, 1]
        self.layers_hf_nl = layers_hf_nl or [3, 10, 10, 1]
        self.layers_hf_l = layers_hf_l or [3, 1]
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2_reg = l2_reg
        self.lf_pretrain_patience = lf_pretrain_patience
        self.verbose = verbose

        self.trainer = None
        self.is_trained = False

    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray) -> Dict[str, Any]:
        """
        Train the MF-DNN on pre-normalized data.

        Args:
            X_lf: LF inputs (N_L, D) — pre-normalized to [-1, 1]
            Y_lf: LF outputs (N_L, 1) — pre-normalized to [-1, 1]
            X_hf: HF inputs (N_H, D) — pre-normalized to [-1, 1]
            Y_hf: HF outputs (N_H, 1) — pre-normalized to [-1, 1]

        Returns:
            Training info dict
        """
        X_lf = np.asarray(X_lf, dtype=np.float32)
        Y_lf = np.asarray(Y_lf, dtype=np.float32).reshape(-1, 1)
        X_hf = np.asarray(X_hf, dtype=np.float32)
        Y_hf = np.asarray(Y_hf, dtype=np.float32).reshape(-1, 1)

        self.trainer = MFTrainer(
            self.layers_lf, self.layers_hf_nl, self.layers_hf_l,
            learning_rate=self.learning_rate, l2_reg=self.l2_reg
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        self.trainer.optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

        x_lf_t = tf.convert_to_tensor(X_lf, dtype=tf.float32)
        y_lf_t = tf.convert_to_tensor(Y_lf, dtype=tf.float32)
        x_hf_t = tf.convert_to_tensor(X_hf, dtype=tf.float32)
        y_hf_t = tf.convert_to_tensor(Y_hf, dtype=tf.float32)

        # ── Phase 1: LF Pretraining ───────────────────────────────────────
        if self.lf_pretrain_patience > 0:
            best_lf = float('inf')
            wait_lf = 0
            best_lf_weights = None
            for epoch in range(self.max_epochs):
                loss_lf = self.trainer.pretrain_step_lf(x_lf_t, y_lf_t)
                val = float(loss_lf)
                if val < best_lf:
                    best_lf = val
                    wait_lf = 0
                    best_lf_weights = [w.numpy() for w in self.trainer.W_lf + self.trainer.b_lf]
                else:
                    wait_lf += 1
                    if wait_lf >= self.lf_pretrain_patience:
                        if self.verbose and epoch % 5000 == 0:
                            import time
                            print(f"Epoch {epoch}: loss={loss_val:.6f} | time={time.strftime('%H:%M:%S')}")
                        break
            if best_lf_weights is not None:
                for w, v in zip(self.trainer.W_lf + self.trainer.b_lf, best_lf_weights):
                    w.assign(v)

        # ── Phase 2: Joint Fine-tuning ────────────────────────────────────
        best_loss = float('inf')
        wait = 0
        best_weights = None
        training_history = []

        for epoch in range(self.max_epochs):
            loss, loss_lf, loss_hf = self.trainer.train_step(
                x_lf_t, y_lf_t, x_hf_t, y_hf_t
            )

            loss_val = float(loss)
            training_history.append({
                'epoch': epoch,
                'loss': loss_val,
                'loss_lf': float(loss_lf),
                'loss_hf': float(loss_hf)
            })

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
        Predict HF output (normalized space — caller/wrapper denormalizes).

        Args:
            X: Input locations (N, D) — pre-normalized
            return_std: Whether to return std

        Returns:
            mean: Predicted HF values (normalized)
            std: Zeros placeholder — use DeepEnsemble for real uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        y_hf, _ = self.trainer.predict(X)
        mean = y_hf.numpy()

        if return_std:
            # No native uncertainty; use DeepEnsemble wrapper for real std.
            std = np.full_like(mean, np.nan)
            return mean, std
        return mean, None

    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output (normalized space — caller/wrapper denormalizes)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        _, y_lf = self.trainer.predict(X)
        return y_lf.numpy()


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing MF-DNN with synthetic data...")

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    np.random.seed(42)

    X_lf = np.random.rand(100, 2).astype(np.float32)
    Y_lf = (np.sin(2 * np.pi * X_lf[:, 0:1]) + 0.1 * np.random.randn(100, 1)).astype(np.float32)

    X_hf = np.random.rand(12, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)

    print("\nTesting MFDNN...")
    model = MFDNN(max_epochs=5000, patience=500, verbose=True)
    info = model.fit(X_lf, Y_lf, X_hf, Y_hf)
    print(f"\nFinal loss: {info['final_loss']:.6f}")
    print(f"Epochs trained: {info['epochs_trained']}")

    X_test = np.random.rand(5, 2).astype(np.float32)
    y_pred, y_std = model.predict(X_test)
    print(f"\nPredictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred.flatten()[:3]}")

    print("\n✓ MF-DNN tests passed!")
