"""
Hybrid KAN+DNN Multi-Fidelity Model

Architecture:
    LF:  real KAN (B-spline, from mf_kan.py)  →  X → Y_lf
    HF nonlinear: real DNN (tanh, from mf_dnn.py) →  [X, Y_lf] → Y_hf_nl
    HF linear:    simple linear layer            →  [X, Y_lf] → Y_hf_l
    Final:        Y_hf = Y_hf_nl + Y_hf_l

Combines KAN interpretability for LF surrogate with DNN flexibility for HF correction.
Normalization handled externally by NormalizingModelWrapper.
"""

import sys
import os
os.environ['TF_DISABLE_METAL'] = '1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Allow both `python -m models.mf_hybrid` and direct execution
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from models.mf_kan import KAN
from models.mf_dnn import DNN


class MFHybridTrainer(tf.Module):
    """
    Multi-Fidelity Hybrid Trainer: KAN for LF, DNN for HF.

    Architecture:
    - LF KAN:          X         → Y_lf
    - HF nonlinear DNN: [X, Y_lf] → Y_hf_nl
    - HF linear:        [X, Y_lf] → Y_hf_l
    - Final HF:         Y_hf = Y_hf_nl + Y_hf_l
    """

    def __init__(self, layers_lf: List[int], layers_hf_nl: List[int],
                 layers_hf_l: List[int], grid_size: int = 5,
                 spline_order: int = 3, learning_rate: float = 0.001,
                 l2_reg: float = 0.01):
        super().__init__()

        # LF: real KAN with B-spline activations
        self.kan_lf = KAN(layers_lf, grid_size, spline_order, name='kan_lf')

        # HF nonlinear: real DNN with tanh activations
        self.dnn = DNN()
        self.W_hf_nl, self.b_hf_nl = self.dnn.hyper_initial(layers_hf_nl)

        # HF linear: simple linear projection
        self.W_hf_l = tf.Variable(
            tf.random.normal([layers_hf_l[0], layers_hf_l[-1]], stddev=0.1),
            name='W_hf_l', dtype=tf.float32
        )
        self.b_hf_l = tf.Variable(
            tf.zeros([layers_hf_l[-1]]),
            name='b_hf_l', dtype=tf.float32
        )

        self.l2_reg = l2_reg
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.lf_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return (
            self.kan_lf.trainable_variables
            + self.W_hf_nl + self.b_hf_nl
            + [self.W_hf_l, self.b_hf_l]
        )

    @tf.function
    def train_step(self, x_lf: tf.Tensor, y_lf: tf.Tensor,
                   x_hf_coords: tf.Tensor, y_hf: tf.Tensor
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Single training step. All inputs pre-normalized to [-1, 1]."""
        with tf.GradientTape() as tape:
            # LF prediction at LF locations
            y_pred_lf = self.kan_lf(x_lf)

            # LF prediction at HF coordinates (for augmentation)
            y_lf_at_hf = self.kan_lf(x_hf_coords)

            # Augment HF coordinates with LF predictions
            x_aug = tf.concat([x_hf_coords, y_lf_at_hf], axis=1)

            # HF predictions: DNN nonlinear + linear
            y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug)
            y_pred_hf_l = tf.matmul(x_aug, self.W_hf_l) + self.b_hf_l
            y_pred_hf = y_pred_hf_nl + y_pred_hf_l

            # Losses
            loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_lf))
            loss_hf = tf.reduce_mean(tf.square(y_pred_hf - y_hf))
            reg_kan = self.kan_lf.regularization_loss(0.001)
            reg_dnn = self.l2_reg * tf.add_n([tf.nn.l2_loss(w) for w in self.W_hf_nl])
            loss = loss_lf + loss_hf + reg_kan + reg_dnn

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, loss_lf, loss_hf

    @tf.function
    def pretrain_step_lf(self, x_lf: tf.Tensor, y_lf: tf.Tensor) -> tf.Tensor:
        """Single LF-only training step (Phase 1 pretraining)."""
        lf_vars = self.kan_lf.trainable_variables
        with tf.GradientTape() as tape:
            y_pred_lf = self.kan_lf(x_lf)
            loss_lf = (tf.reduce_mean(tf.square(y_pred_lf - y_lf))
                       + self.kan_lf.regularization_loss(0.001))
        grads = tape.gradient(loss_lf, lf_vars)
        self.lf_optimizer.apply_gradients(zip(grads, lf_vars))
        return loss_lf

    def predict(self, x_coords: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict LF and HF. Inputs pre-normalized; outputs in normalized space."""
        y_pred_lf = self.kan_lf(x_coords)
        x_aug = tf.concat([x_coords, y_pred_lf], axis=1)
        y_pred_hf_nl = self.dnn.fnn(self.W_hf_nl, self.b_hf_nl, x_aug)
        y_pred_hf_l = tf.matmul(x_aug, self.W_hf_l) + self.b_hf_l
        y_pred_hf = y_pred_hf_nl + y_pred_hf_l
        return y_pred_hf, y_pred_lf


class HybridKANDNN:
    """
    Hybrid Multi-Fidelity Model: KAN for LF + DNN for HF correction.

    Clean interface matching GP/DNN/KAN API.
    Expects pre-normalized data (handled by NormalizingModelWrapper).

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
                 l2_reg: float = 0.01,
                 lf_pretrain_patience: int = 500,
                 verbose: bool = True):
        self.name = "Hybrid-KAN-DNN"
        self.layers_lf = layers_lf or [2, 20, 20, 1]
        self.layers_hf_nl = layers_hf_nl or [3, 10, 10, 1]
        self.layers_hf_l = layers_hf_l or [3, 1]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2_reg = l2_reg
        self.lf_pretrain_patience = lf_pretrain_patience
        self.verbose = verbose

        self.trainer = None
        self.is_trained = False

    def fit(self, X_lf: np.ndarray, Y_lf: np.ndarray,
            X_hf: np.ndarray, Y_hf: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the hybrid model on pre-normalized data.

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

        self.trainer = MFHybridTrainer(
            self.layers_lf, self.layers_hf_nl, self.layers_hf_l,
            grid_size=self.grid_size, spline_order=self.spline_order,
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

        # ── Phase 1: LF KAN Pretraining ──────────────────────────────────
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
                    best_lf_weights = [v.numpy() for v in self.trainer.kan_lf.trainable_variables]
                else:
                    wait_lf += 1
                    if wait_lf >= self.lf_pretrain_patience:
                        if self.verbose:
                            print(f"LF pretrain done at epoch {epoch}, loss_lf={best_lf:.6f}")
                        break
            if best_lf_weights is not None:
                for v, val in zip(self.trainer.kan_lf.trainable_variables, best_lf_weights):
                    v.assign(val)

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
                best_weights = [v.numpy() for v in self.trainer.trainable_variables]
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if self.verbose and epoch % 5000 == 0:
                print(f"Epoch {epoch}: loss={loss_val:.6f}, "
                      f"LF={float(loss_lf):.6f}, HF={float(loss_hf):.6f}")

        if best_weights is not None:
            for v, val in zip(self.trainer.trainable_variables, best_weights):
                v.assign(val)

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
            std: Zeros placeholder (no MC Dropout in this architecture)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_t = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        y_hf, _ = self.trainer.predict(X_t)
        mean = y_hf.numpy()

        if return_std:
            # No native uncertainty; use DeepEnsemble wrapper for real std.
            std = np.zeros_like(mean)
            return mean, std
        return mean, None

    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """Predict LF output from KAN (normalized space — caller denormalizes)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_t = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        _, y_lf = self.trainer.predict(X_t)
        return y_lf.numpy()

    def get_learned_rho(self) -> Optional[float]:
        """Not applicable for this architecture (no rho scaling). Returns None."""
        return None


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing Hybrid KAN+DNN model...")

    import time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    np.random.seed(42)

    X_lf = np.random.rand(100, 2).astype(np.float32)
    Y_lf = (np.sin(2 * np.pi * X_lf[:, 0:1]) + 0.1 * np.random.randn(100, 1)).astype(np.float32)

    X_hf = np.random.rand(12, 2).astype(np.float32)
    Y_hf = (np.sin(2 * np.pi * X_hf[:, 0:1]) + 0.5 * X_hf[:, 1:2]).astype(np.float32)

    model = HybridKANDNN(max_epochs=3000, patience=300, verbose=True)

    t_fit_start = time.time()
    info = model.fit(X_lf, Y_lf, X_hf, Y_hf)
    fit_elapsed = time.time() - t_fit_start

    print(f"\nFinal loss: {info['final_loss']:.6f}")
    print(f"Epochs trained: {info['epochs_trained']}")
    print(f"Fit time: {fit_elapsed:.2f}s ({info['epochs_trained']/fit_elapsed:.1f} epochs/sec)")

    X_test = np.random.rand(5, 2).astype(np.float32)

    t_pred_start = time.time()
    y_pred, y_std = model.predict(X_test, return_std=True)
    pred_elapsed = time.time() - t_pred_start

    print(f"\nPredictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred.flatten()[:3]}")
    print(f"Predict time: {pred_elapsed*1000:.2f}ms")

    y_lf = model.predict_lf(X_test)
    print(f"LF predictions: {y_lf.flatten()[:3]}")

    print("\nHybrid KAN+DNN test passed!")
