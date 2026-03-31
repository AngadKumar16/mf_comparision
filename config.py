"""
Configuration for Multi-Fidelity Comparison Project
"""
from pathlib import Path
 
# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
 
# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
 
# Your local data path (update this to your actual path)
MATLAB_DATA_PATH = "/Users/angadkumar16ak/Shukla/Data/Data/Data.mat"
 
# ============================================================
# DATA SPLIT CONFIGURATION
# ============================================================
N_HF_TRAIN = 12  # Number of HF points for training
N_HF_TEST = 2    # Remaining HF points for testing (14 - 12 = 2)
RANDOM_SEED = 42
 
# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
 
# Gaussian Process
GP_CONFIG = {
    'kernel': 'RBF',
    'ARD': True,
    'num_restarts': 6,
}
 
# Deep Neural Network
DNN_CONFIG = {
    'layers_lf': [2, 20, 20, 1],      # Input: 2D coords, Output: 1D
    'layers_hf_nl': [3, 10, 10, 1],   # Input: 2D + LF pred = 3D
    'layers_hf_l': [3, 1],            # Linear layer
    'learning_rate': 0.001,
    'max_epochs': 30000,
    'patience': 2000,
    'l2_reg': 0.01,
}
 
# Kolmogorov-Arnold Network
KAN_CONFIG = {
    'layers_lf': [2, 20, 20, 1],
    'layers_hf_nl': [3, 10, 10, 1],
    'layers_hf_l': [3, 1],
    'grid_size': 5,
    'spline_order': 3,
    'learning_rate': 0.001,
    'max_epochs': 30000,
    'patience': 1000,
}
 
# Hybrid KAN+DNN (NOVELTY)
HYBRID_CONFIG = {
    'kan_layers': [2, 20, 20, 1],
    'mlp_layers': [3, 32, 32, 1],
    'kan_grid_size': 5,
    'kan_spline_order': 3,
    'dropout_rate': 0.1,
}
 
# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_NOISE_TRIALS = 5
N_ENSEMBLE_MEMBERS = 5
N_MC_SAMPLES = 100  # For MC Dropout uncertainty
 
# ============================================================
# VISUALIZATION
# ============================================================
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 150,
    'cmap': 'rainbow',
    'temp_range': (17, 25),  # Temperature range for colorbar
}
