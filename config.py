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
MATLAB_DATA_PATH = str(PROJECT_ROOT / "data" / "real" / "Data" / "Data.mat")
 
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
    'num_restarts': 10,
}
 
## ============================================================
# SHARED TRAINING BUDGET
# ============================================================
MAX_EPOCHS = 10000          # Safety ceiling; early stopping does the real work
LF_PRETRAIN_PATIENCE = 500  # Phase 1: stop LF pretraining after no improvement
JOINT_PATIENCE = 2000       # Phase 2: stop joint training after no improvement

# Deep Neural Network
DNN_CONFIG = {
    'layers_lf': [2, 20, 20, 1],
    'layers_hf_nl': [3, 8, 1],
    'layers_hf_l': [3, 1],
    'learning_rate': 0.001,
    'max_epochs': MAX_EPOCHS,
    'patience': JOINT_PATIENCE,
    'l2_reg': 0.01,
    'lf_pretrain_patience': LF_PRETRAIN_PATIENCE,
}

# Kolmogorov-Arnold Network
KAN_CONFIG = {
    'layers_lf': [2, 20, 20, 1],
    'layers_hf_nl': [3, 10, 10, 1],
    'layers_hf_l': [3, 1],
    'grid_size': 5,
    'spline_order': 3,
    'learning_rate': 0.001,
    'max_epochs': MAX_EPOCHS,
    'patience': JOINT_PATIENCE,
    'lf_pretrain_patience': LF_PRETRAIN_PATIENCE,
}

# Hybrid KAN+DNN (NOVELTY)
HYBRID_CONFIG = {
    'layers_lf': [2, 20, 20, 1],
    'layers_hf_nl': [3, 32, 32, 1],
    'layers_hf_l': [3, 1],
    'grid_size': 5,
    'spline_order': 3,
    'learning_rate': 0.001,
    'l2_reg': 0.01,
    'max_epochs': MAX_EPOCHS,
    'patience': JOINT_PATIENCE,
    'lf_pretrain_patience': LF_PRETRAIN_PATIENCE,
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
