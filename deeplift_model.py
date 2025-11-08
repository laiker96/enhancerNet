import torch
import numpy as np
from captum.attr import DeepLift
from tqdm import tqdm

def one_hot_encoding(sequence: str, num_mutations: int = 0) -> np.ndarray:
    """
    Encode a DNA sequence to a one-hot encoded matrix and add N mutations during encoding.

    Parameters:
        - sequence (str): DNA sequence to be encoded.
        - num_mutations (int): Number of random mutations to introduce.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (4 x seq_length).
                N's are treated as a uniform distribution (0.25 in all 4 channels that sum to 1).
    """

    # Define nucleotide mapping and initialization
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)
    sequence_upper = sequence.upper()

    # Convert sequence to integer indices
    indices = np.array([mapping.get(base, 4) for base in sequence_upper])

    # Apply base encoding for A, C, G, T, and N
    valid_indices = indices[indices < 4]  # Exclude 'N' positions
    encoding[valid_indices, np.arange(seq_len)[indices < 4]] = 1.0

    # Handle 'N' positions
    n_positions = (indices == 4)
    encoding[:, n_positions] = 0.25

    # Mutation Handling
    if num_mutations > 0:
        # Choose mutation positions and mutation types
        mutation_positions = np.random.choice(seq_len, size=num_mutations, replace=False)
        mutation_types = np.random.choice(4, size=num_mutations)

        # Apply mutations
        encoding[:, mutation_positions] = 0  # Reset positions to zero
        encoding[mutation_types, mutation_positions] = 1.0  # Apply mutations

    return encoding

def dinucleotide_shuffle(one_hot_sequence):
    """
    Shuffles dinucleotides in a one-hot encoded sequence.
    
    Parameters:
    - one_hot_sequence (numpy array): Shape (4, L), where L is the sequence length,
      and the one-hot encoding has 4 rows representing the nucleotides A, C, G, T.
    
    Returns:
    - shuffled_sequence (numpy array): Dinucleotide-shuffled version of the input sequence,
      with the same shape as the input (4, L).
    """
    # Convert one-hot encoding to nucleotide sequence (e.g., 'A', 'C', 'G', 'T')
    nucleotide_map = ['A', 'C', 'G', 'T']
    seq_len = one_hot_sequence.shape[1]
    
    # Decode the one-hot sequence to get the original nucleotide string
    decoded_sequence = ''.join([nucleotide_map[np.argmax(one_hot_sequence[:, i])] for i in range(seq_len)])
    
    # Split the sequence into dinucleotides
    dinucleotides = [decoded_sequence[i:i+2] for i in range(0, seq_len, 2)]
    
    # Shuffle the dinucleotides
    np.random.shuffle(dinucleotides)
    
    # Join the shuffled dinucleotides back into a string
    shuffled_sequence_str = ''.join(dinucleotides)
    
    # Reconvert the shuffled sequence back to one-hot encoding
    
    return one_hot_encoding(shuffled_sequence_str, num_mutations=0)

def compute_deeplift_attributions2(
    model,
    input_seqs,
    dinucleotide_shuffle_fn,
    target_class=8,
    num_baselines=10
):
    model.eval()
    deeplift = DeepLift(model)

    N, C, L = input_seqs.shape
    all_attributions = torch.zeros_like(input_seqs, device=input_seqs.device)

    for i, input_seq in enumerate(tqdm(input_seqs, desc="Computing DeepLift attributions")):
        # Shape: (1, C, L)
        input_seq = input_seq.unsqueeze(0).to(input_seqs.device)
        input_seq.requires_grad_()

        # Convert to NumPy once
        seq_np = input_seq[0].detach().cpu().numpy()

        # Generate all baselines for this sequence at once
        shuffled_list = [
            torch.tensor(dinucleotide_shuffle_fn(seq_np), device=input_seq.device)
            for _ in range(num_baselines)
        ]
        baselines = torch.stack(shuffled_list)  # shape: (num_baselines, C, L)

        # Repeat the same input sequence for each baseline
        inputs_repeated = input_seq.repeat(num_baselines, 1, 1)  # shape: (num_baselines, C, L)

        # One DeepLift call for all baselines
        attributions = deeplift.attribute(
            inputs=inputs_repeated,
            baselines=baselines,
            target=target_class
        )  # shape: (num_baselines, C, L)

        # Average over baselines
        attributions_avg = attributions.mean(dim=0, keepdim=True)  # shape: (1, C, L)

        # Store result
        all_attributions[i] = attributions_avg.squeeze(0)

    print("Shape of averaged attributions:", all_attributions.shape)
    return all_attributions
