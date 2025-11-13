import pybedtools as pybed
import os
import argparse
import numpy as np
from npy_append_array import NpyAppendArray

def get_sequences(bed_file_name: str, fasta_file: str) -> str:
    """
    Extract sequences from a BED file and save them in a FASTA file.
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
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)

    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_upper = np.where((seq_bytes >= 97) & (seq_bytes <= 122), seq_bytes - 32, seq_bytes)

    lut = np.full(128, 4, dtype=np.uint8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3

    indices = lut[seq_upper]
    valid_mask = indices < 4
    valid_positions = np.nonzero(valid_mask)[0]
    encoding[indices[valid_mask], valid_positions] = 1.0

    if np.any(~valid_mask):
        encoding[:, ~valid_mask] = 0.25

    return encoding


def process_fasta(filename: str) -> str:
    """
    Process a multi-FASTA file into a numpy array of one-hot encoded sequences.
    Returns the output .npy filename.
    """
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

    return output_name


def main(input_file: str, genome_file: str = None) -> str:
    """
    Main function to process either a BED or FASTA file, encoding sequences into .npy format.

    Parameters:
    - input_file (str): Path to the BED or FASTA file.
    - genome_file (str, optional): Path to the genome FASTA file (required for BED input).

    Returns:
    - str: Path to the generated .npy file.
    """
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext in ['.bed']:
        if genome_file is None:
            raise ValueError("A genome FASTA file must be provided when using a BED input.")
        fasta_filename = get_sequences(input_file, genome_file)
        npy_filename = process_fasta(fasta_filename)
    elif file_ext in ['.fa', '.fasta']:
        npy_filename = process_fasta(input_file)
    else:
        raise ValueError("Unsupported file type. Please provide a .bed, .fa, or .fasta file.")

    print(f"Encoded file saved to: {npy_filename}")
    return npy_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process BED or FASTA file and generate one-hot encoded numpy array.'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input BED or FASTA file.')
    parser.add_argument('--genome', type=str, required=False,
                        help='Path to the genome FASTA file (required if input is BED).')

    args = parser.parse_args()
    main(args.input, args.genome)