import numpy as np
import pandas as pd

# Step 1: Load the .npy target array
targets = np.load("train_target.npy")  # shape: (1286550, 9)
assert targets.ndim == 2 and targets.shape[1] == 9, "Expected shape (N, 9)"

# Step 2: Compute column-wise mean and std
means = targets.mean(axis=0)
stds = targets.std(axis=0)

# Step 3: Z-score normalization
zscore_targets = (targets - means) / stds

# Step 4: Save normalized targets
np.save("targets_zscore.npy", zscore_targets)

# Step 5: Save mean and std for inverse-transform
np.save("zscore_means.npy", means)
np.save("zscore_stds.npy", stds)
print("✅ Z-score normalization complete and saved.")
