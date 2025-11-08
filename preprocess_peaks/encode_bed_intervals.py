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
    Encode a DNA sequence to a one-hot encoded matrix.

    Parameters:
        - sequence (str): DNA sequence to be encoded.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (4 x seq_length).
        N's are treated as a uniform distribution (0.25 in all 4 channels that sum to 1).
    """
        
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = np.zeros((4, len(sequence)), dtype = np.float32)
    sequence_upper = sequence.upper()
        
    for i, base in enumerate(sequence_upper):
        if base in mapping:
            encoding[mapping[base], i] = 1.0
        elif base == 'N':
            encoding[:, i] = 0.25 
    return encoding.reshape(1, 4, -1)

def process_fasta(filename: str) -> None:
    """
    Process a multi-FASTA file into a numpy array of one-hot encoded sequences.

    Parameters:
    - filename (str): Path to the input multi-FASTA file.

    Returns:
    - None
    """

    file_basename = os.path.splitext(filename)[0]                                             
    output_name = file_basename + '_encoding.npy'

    with open(filename, 'rt') as file:
        lines = file.readlines()

    with NpyAppendArray(output_name, delete_if_exists=True) as output:
        for line in lines:
            if line.startswith('>'):
                continue
            else:
                sequence = np.array([line.strip()])
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
