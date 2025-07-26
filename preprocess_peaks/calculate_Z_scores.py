import argparse
import numpy as np

def zscore_normalize(
    input_npy,
    output_npy="targets_zscore.npy",
    mean_file="zscore_means.npy",
    std_file="zscore_stds.npy"
):
    # Load input
    targets = np.load(input_npy)
    assert targets.ndim == 2, f"Expected 2D array, got shape {targets.shape}"

    # Compute mean and std
    means = targets.mean(axis=0)
    stds = targets.std(axis=0)

    # Z-score normalization
    zscore_targets = (targets - means) / stds

    # Save outputs
    np.save(output_npy, zscore_targets)
    np.save(mean_file, means)
    np.save(std_file, stds)

    print("✅ Z-score normalization complete and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-score normalize a target .npy array")
    parser.add_argument("input_npy", help="Input .npy file (2D array)")
    parser.add_argument("--output", "-o", default="targets_zscore.npy", help="Normalized output .npy file")
    parser.add_argument("--mean", "-m", default="zscore_means.npy", help="File to save means")
    parser.add_argument("--std", "-s", default="zscore_stds.npy", help="File to save stds")

    args = parser.parse_args()
    zscore_normalize(args.input_npy, args.output, args.mean, args.std)

