# diagnose_forrester.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic.forrester import Forrester2D
data = Forrester2D.generate_data()

print("Forrester data ranges:")
print(f"  X_lf:        shape={data['X_lf'].shape}")
print(f"  Y_lf range:  [{data['Y_lf'].min():.3f}, {data['Y_lf'].max():.3f}]")
print(f"  X_hf_train:  shape={data['X_hf_train'].shape}")
print(f"  Y_hf_train:  [{data['Y_hf_train'].min():.3f}, {data['Y_hf_train'].max():.3f}]")

# Check normalization extremes
y_lf_min = data['Y_lf'].min()
y_lf_max = data['Y_lf'].max()
yr = y_lf_max - y_lf_min

y_hf_norm = 2 * (data['Y_hf_train'] - y_lf_min) / yr - 1
print(f"\n  Y_hf after LF-based normalization:")
print(f"    Range: [{y_hf_norm.min():.3f}, {y_hf_norm.max():.3f}]")
print(f"    Outside [-1, 1]: {(abs(y_hf_norm) > 1).sum()} of {y_hf_norm.size} points")