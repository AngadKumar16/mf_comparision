"""
Synthetic Test Functions for Multi-Fidelity Validation

Provides:
1. Forrester2D: 2D extension of classic Forrester function
2. Branin2D: Multi-fidelity Branin function

Use these for debugging before running on real data.
"""

import numpy as np
from typing import Dict, Any, Tuple


class Forrester2D:
    """
    2D Forrester function for multi-fidelity testing.
    
    The classic 1D Forrester function is:
        f(x) = (6x - 2)² sin(12x - 4)
    
    We extend to 2D with:
        f(x,y) = f_x + f_y + interaction
    """
    
    @staticmethod
    def high_fidelity(X: np.ndarray) -> np.ndarray:
        """
        True HF function.
        
        Args:
            X: Input array (N, 2) with values in [0, 1]
        
        Returns:
            Y: Output array (N, 1)
        """
        X = np.atleast_2d(X)
        x, y = X[:, 0], X[:, 1]
        
        # 1D Forrester components
        f_x = (6 * x - 2)**2 * np.sin(12 * x - 4)
        f_y = (6 * y - 2)**2 * np.sin(12 * y - 4)
        
        # Interaction term
        interaction = 2 * np.sin(5 * x * y)
        
        return (f_x + f_y + interaction).reshape(-1, 1)
    
    @staticmethod
    def low_fidelity(X: np.ndarray) -> np.ndarray:
        """
        LF approximation: linear transform + bias.
        
        f_lf(x) = A * f_hf(x) + B * (x - 0.5) + C
        """
        X = np.atleast_2d(X)
        
        A, B, C = 0.5, 10, -5  # Standard scaling
        
        hf = Forrester2D.high_fidelity(X)
        bias = B * (X[:, 0:1] - 0.5) + B * (X[:, 1:2] - 0.5)
        
        return A * hf + bias + C
    
    @staticmethod
    def generate_data(n_lf: int = 200, 
                      n_hf_train: int = 12,
                      n_hf_test: int = 50,
                      noise_lf: float = 0.0,
                      noise_hf: float = 0.0,
                      seed: int = 42) -> Dict[str, Any]:
        """
        Generate train/test data matching real data setup.
        
        Args:
            n_lf: Number of LF points
            n_hf_train: Number of HF training points
            n_hf_test: Number of HF test points
            noise_lf: Noise std for LF data
            noise_hf: Noise std for HF data
            seed: Random seed
        
        Returns:
            Dict with X_lf, Y_lf, X_hf_train, Y_hf_train, X_hf_test, Y_hf_test
        """
        np.random.seed(seed)
        
        try:
            from pyDOE import lhs
            # Latin Hypercube Sampling for better coverage
            X_lf = lhs(2, samples=n_lf)
            X_hf_all = lhs(2, samples=n_hf_train + n_hf_test)
        except ImportError:
            # Fallback to random
            X_lf = np.random.rand(n_lf, 2)
            X_hf_all = np.random.rand(n_hf_train + n_hf_test, 2)
        
        # Generate outputs
        Y_lf = Forrester2D.low_fidelity(X_lf)
        Y_hf_all = Forrester2D.high_fidelity(X_hf_all)
        
        # Add noise
        if noise_lf > 0:
            Y_lf += noise_lf * np.random.randn(*Y_lf.shape)
        if noise_hf > 0:
            Y_hf_all += noise_hf * np.random.randn(*Y_hf_all.shape)
        
        # Split HF
        X_hf_train = X_hf_all[:n_hf_train]
        Y_hf_train = Y_hf_all[:n_hf_train]
        X_hf_test = X_hf_all[n_hf_train:]
        Y_hf_test = Y_hf_all[n_hf_train:]
        
        return {
            'X_lf': X_lf.astype(np.float32),
            'Y_lf': Y_lf.astype(np.float32),
            'X_hf_train': X_hf_train.astype(np.float32),
            'Y_hf_train': Y_hf_train.astype(np.float32),
            'X_hf_test': X_hf_test.astype(np.float32),
            'Y_hf_test': Y_hf_test.astype(np.float32),
            'hf_func': Forrester2D.high_fidelity,
            'lf_func': Forrester2D.low_fidelity,
        }


class Branin2D:
    """
    Multi-fidelity Branin function.
    
    Classic Branin (HF):
        f(x1, x2) = a*(x2 - b*x1² + c*x1 - r)² + s*(1-t)*cos(x1) + s
    
    where:
        a = 1, b = 5.1/(4π²), c = 5/π, r = 6, s = 10, t = 1/(8π)
    
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    """
    
    @staticmethod
    def high_fidelity(X: np.ndarray) -> np.ndarray:
        """Standard Branin function."""
        X = np.atleast_2d(X)
        
        # Rescale from [0,1]² to actual domain
        x1 = X[:, 0] * 15 - 5  # [-5, 10]
        x2 = X[:, 1] * 15      # [0, 15]
        
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        
        return (term1 + term2 + s).reshape(-1, 1)
    
    @staticmethod
    def low_fidelity(X: np.ndarray) -> np.ndarray:
        """LF approximation with simplified coefficients."""
        X = np.atleast_2d(X)
        
        x1 = X[:, 0] * 15 - 5
        x2 = X[:, 1] * 15
        
        # Modified coefficients (cheaper computation analogy)
        a = 0.9
        b = 5.0 / (4 * np.pi**2)  # Slightly off
        c = 5 / np.pi
        r = 5.5  # Slightly off
        s = 10
        t = 1 / (8 * np.pi)
        
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        
        return (term1 + term2 + s + 5).reshape(-1, 1)  # Bias added
    
    @staticmethod
    def generate_data(n_lf: int = 200, 
                      n_hf_train: int = 12,
                      n_hf_test: int = 50,
                      seed: int = 42) -> Dict[str, Any]:
        """Generate data similar to Forrester2D."""
        np.random.seed(seed)
        
        try:
            from pyDOE import lhs
            X_lf = lhs(2, samples=n_lf)
            X_hf_all = lhs(2, samples=n_hf_train + n_hf_test)
        except ImportError:
            X_lf = np.random.rand(n_lf, 2)
            X_hf_all = np.random.rand(n_hf_train + n_hf_test, 2)
        
        Y_lf = Branin2D.low_fidelity(X_lf)
        Y_hf_all = Branin2D.high_fidelity(X_hf_all)
        
        X_hf_train = X_hf_all[:n_hf_train]
        Y_hf_train = Y_hf_all[:n_hf_train]
        X_hf_test = X_hf_all[n_hf_train:]
        Y_hf_test = Y_hf_all[n_hf_train:]
        
        return {
            'X_lf': X_lf.astype(np.float32),
            'Y_lf': Y_lf.astype(np.float32),
            'X_hf_train': X_hf_train.astype(np.float32),
            'Y_hf_train': Y_hf_train.astype(np.float32),
            'X_hf_test': X_hf_test.astype(np.float32),
            'Y_hf_test': Y_hf_test.astype(np.float32),
            'hf_func': Branin2D.high_fidelity,
            'lf_func': Branin2D.low_fidelity,
        }


def visualize_test_function(func_class, n_grid: int = 50):
    """Visualize HF and LF surfaces of a test function."""
    import matplotlib.pyplot as plt
    
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    
    Z_hf = func_class.high_fidelity(XY).reshape(n_grid, n_grid)
    Z_lf = func_class.low_fidelity(XY).reshape(n_grid, n_grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # HF
    im1 = axes[0].contourf(X, Y, Z_hf, levels=20, cmap='viridis')
    axes[0].set_title('High-Fidelity')
    plt.colorbar(im1, ax=axes[0])
    
    # LF
    im2 = axes[1].contourf(X, Y, Z_lf, levels=20, cmap='viridis')
    axes[1].set_title('Low-Fidelity')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    im3 = axes[2].contourf(X, Y, Z_hf - Z_lf, levels=20, cmap='coolwarm')
    axes[2].set_title('HF - LF (Discrepancy)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(func_class.__name__)
    plt.tight_layout()
    plt.show()


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing synthetic functions...")
    
    # Test Forrester2D
    print("\n1. Forrester2D")
    data = Forrester2D.generate_data(n_lf=100, n_hf_train=12, n_hf_test=20)
    print(f"   LF: {data['X_lf'].shape}, HF train: {data['X_hf_train'].shape}")
    print(f"   Y_lf range: [{data['Y_lf'].min():.2f}, {data['Y_lf'].max():.2f}]")
    print(f"   Y_hf range: [{data['Y_hf_train'].min():.2f}, {data['Y_hf_train'].max():.2f}]")
    
    # Test Branin2D
    print("\n2. Branin2D")
    data = Branin2D.generate_data(n_lf=100, n_hf_train=12, n_hf_test=20)
    print(f"   LF: {data['X_lf'].shape}, HF train: {data['X_hf_train'].shape}")
    print(f"   Y_lf range: [{data['Y_lf'].min():.2f}, {data['Y_lf'].max():.2f}]")
    print(f"   Y_hf range: [{data['Y_hf_train'].min():.2f}, {data['Y_hf_train'].max():.2f}]")
    
    # Visualize (optional)
    try:
        print("\n3. Visualizing Forrester2D...")
        visualize_test_function(Forrester2D)
    except Exception as e:
        print(f"   Visualization skipped: {e}")
    
    print("\n✓ Synthetic function tests passed!")