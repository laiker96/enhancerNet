#!/usr/bin/env python
import argparse
import numpy as np

def one_hot_encode_sequence(seq, target_len=1000):
    """
    One-hot encode a DNA sequence (A,C,G,T) into shape (4, target_len).
    Unknown characters are zeros.
    Sequences longer than target_len are truncated.
    Sequences shorter are zero-padded.
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    seq = seq.upper()
    encoding = np.zeros((4, target_len), dtype=np.float32)
    
    for i, base in enumerate(seq[:target_len]):  # truncate if longer
        if base in mapping:
            encoding[mapping[base], i] = 1.0
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

    # Convert to numpy array
    encoded_array = np.stack(sequences, axis=0)  # shape: [batch_size, 4, 1000]
    np.save(args.output, encoded_array)
    print(f"Saved encoded array with shape {encoded_array.shape} to {args.output}")

if __name__ == "__main__":
    main()

