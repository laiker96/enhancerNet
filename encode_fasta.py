#!/usr/bin/env python
import argparse
import numpy as np

def one_hot_encode_sequence(seq):
    """
    One-hot encode a DNA sequence (A,C,G,T) into shape (length, 4)
    Unknown characters are all zeros.
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    seq_len = len(seq)
    encoding = np.zeros((seq_len, 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            encoding[i, mapping[base]] = 1.0
    return encoding

def main():
    parser = argparse.ArgumentParser(description="One-hot encode DNA sequences from a FASTA file")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Output npy file for encoded sequences")
    args = parser.parse_args()

    sequences = []
    with open(args.fasta, 'r') as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(one_hot_encode_sequence(seq))
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(one_hot_encode_sequence(seq))

    # Pad sequences to same length if needed
    max_len = max(s.shape[0] for s in sequences)
    encoded_array = np.zeros((len(sequences), max_len, 4), dtype=np.float32)
    for i, s in enumerate(sequences):
        encoded_array[i, :s.shape[0], :] = s

    np.save(args.output, encoded_array)

if __name__ == "__main__":
    main()
