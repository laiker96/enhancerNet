import argparse
import pandas as pd
import numpy as np

def extract_target(input_file, number_of_samples=9, output_name="target.npy"):
    dataframe = pd.read_csv(input_file, sep="\t", header=None)
    output = np.array(dataframe.iloc[:, -number_of_samples:])
    np.save(output_name, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract target columns and save as .npy")
    parser.add_argument("input_file", help="Path to input TSV file")
    parser.add_argument("--samples", "-s", type=int, default=9,
                        help="Number of target columns to extract from the end (default: 9)")
    parser.add_argument("--output", "-o", default="target.npy",
                        help="Output .npy filename (default: target.npy)")

    args = parser.parse_args()
    extract_target(args.input_file, args.samples, args.output)
