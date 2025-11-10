import pybedtools as pybed
import os
import argparse
import numpy as np
from npy_append_array import NpyAppendArray

def get_sequences(bed_file_name: str, fasta_file: str) -> str:
    """
    Extract sequences from a BED file and save them in a FASTA file.

    Parameters:
    - bed_file_name (str): Path to the input BED file.
    - fasta_file (str): Path to the reference FASTA file.

    Returns:
    - str: Path to the generated FASTA file.
    """
    bed_file = pybed.BedTool(bed_file_name)
    fasta_filename = os.path.splitext(bed_file_name)[0] + '.fa'
    bed_file.sequence(fi=fasta_file, s=True)
    bed_file.save_seqs(fasta_filename)
    return fasta_filename

def one_hot_encoding(sequence: str) -> np.ndarray:
    """
    Encode a DNA sequence into a 4×L one-hot matrix.
    'A', 'C', 'G', 'T' → one-hot channels
    'N' or unknown → uniform [0.25, 0.25, 0.25, 0.25]
    """
    # Preallocate output: (4 × L)
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)

    # Fastest possible uppercase ASCII conversion
    # (no Python loop, works directly on bytes)
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_upper = np.where((seq_bytes >= 97) & (seq_bytes <= 122),  # a-z → A-Z
                         seq_bytes - 32, seq_bytes)

    # Lookup table for ASCII → channel index (A,C,G,T = 0–3, others = 4)
    lut = np.full(128, 4, dtype=np.uint8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3

    # Vectorized map to indices (0–4)
    indices = lut[seq_upper]

    # Boolean mask for valid bases
    valid_mask = indices < 4
    valid_positions = np.nonzero(valid_mask)[0]

    # Efficient scatter using advanced indexing
    encoding[indices[valid_mask], valid_positions] = 1.0

    # Assign 0.25 for 'N' and unknown bases (indices == 4)
    if np.any(~valid_mask):
        encoding[:, ~valid_mask] = 0.25

    return encoding

def process_fasta(filename: str) -> None:
    """
    Process a multi-FASTA file into a numpy array of one-hot encoded sequences.

    Each sequence becomes shape (1, L, 4), so appending produces (N, L, 4).
    """
    import os
    from npy_append_array import NpyAppendArray

    file_basename = os.path.splitext(filename)[0]                                             
    output_name = file_basename + '_encoding.npy'

    with open(filename, 'rt') as file:
        lines = file.readlines()

    with NpyAppendArray(output_name, delete_if_exists=True) as output:
        for line in lines:
            if line.startswith('>'):
                continue

            sequence = one_hot_encoding(line.strip())  
            sequence = np.expand_dims(sequence, axis=0)  

            output.append(sequence)

                

def main(bed_file_name: str, genome_file: str) -> None:
    """
    Main function to process BED and FASTA files, extracting sequences and encoding them.

    Parameters:
    - bed_file_name (str): Path to the BED file.
    - genome_file (str): Path to the genome FASTA file.

    Returns:
    - None
    """
    fasta_filename = get_sequences(bed_file_name, genome_file)
    process_fasta(fasta_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process BED file and generate sequence embeddings.')
    parser.add_argument('--bed_filename', type=str, help='Path to the BED file input.')
    parser.add_argument('--genome_filename', type=str, help='Path to the genome FASTA file.')

    args = parser.parse_args()
    args_list = list(vars(args).values())
    main(*args_list)
