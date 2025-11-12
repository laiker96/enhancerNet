"""
attribution_utils.py
---------------------

DNA sequence utilities and DeepLIFT attribution tools for enhancer models.

Includes:
    - Fast one-hot encoding for DNA sequences
    - Dinucleotide shuffling baseline generator
    - Memory-efficient DeepLIFT computation
    - Pleiotropic DeepLIFT across active tasks

Dependencies:
    numpy, torch, captum, tqdm
"""

import numpy as np
import torch
import torch.nn as nn
from captum.attr import DeepLift, IntegratedGradients
from tqdm import tqdm


# =========================================================
# 🧬 FAST ONE-HOT ENCODING
# =========================================================
def one_hot_encoding(sequence: str) -> np.ndarray:
    """
    Encode a DNA sequence into a 4×L one-hot matrix.
    'A', 'C', 'G', 'T' → one-hot channels
    'N' or unknown → uniform [0.25, 0.25, 0.25, 0.25]
    """
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)

    # Convert to uppercase ASCII bytes
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_upper = np.where((seq_bytes >= 97) & (seq_bytes <= 122),  # a-z → A-Z
                         seq_bytes - 32, seq_bytes)

    # Lookup table for ASCII → index (A,C,G,T = 0–3, others = 4)
    lut = np.full(128, 4, dtype=np.uint8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3

    indices = lut[seq_upper]
    valid_mask = indices < 4
    valid_positions = np.nonzero(valid_mask)[0]

    # Scatter valid positions
    encoding[indices[valid_mask], valid_positions] = 1.0

    # 'N' or unknown → uniform
    if np.any(~valid_mask):
        encoding[:, ~valid_mask] = 0.25

    return encoding


# =========================================================
# 🌀 DINUCLEOTIDE SHUFFLING
# =========================================================
def dinucleotide_shuffle(one_hot_sequence: np.ndarray) -> np.ndarray:
    """
    Dinucleotide-preserving shuffle for a one-hot encoded sequence.

    Args:
        one_hot_sequence (np.ndarray): 4×L one-hot matrix

    Returns:
        np.ndarray: 4×L one-hot matrix after dinucleotide shuffling
    """
    nucleotide_map = ['A', 'C', 'G', 'T']
    seq_len = one_hot_sequence.shape[1]

    # Decode one-hot → string
    decoded = ''.join([nucleotide_map[np.argmax(one_hot_sequence[:, i])] for i in range(seq_len)])

    # Handle odd-length sequences
    if seq_len % 2 != 0:
        decoded = decoded[:-1]
        seq_len -= 1

    # Split into dinucleotides and shuffle
    dinucs = [decoded[i:i+2] for i in range(0, seq_len, 2)]
    np.random.shuffle(dinucs)
    shuffled_seq = ''.join(dinucs)

    return one_hot_encoding(shuffled_seq)


# =========================================================
# ⚡ SINGLE-TASK DEEPLIFT ATTRIBUTION
# =========================================================
def compute_deeplift_attributions2(
    model,
    input_seqs,
    dinucleotide_shuffle_fn,
    target_class=0,
    num_baselines=10,
    cleanup_interval=50,
    device="cuda"
):
    """
    Compute DeepLIFT attributions for a single model output (target_class)
    using multiple dinucleotide-shuffled baselines.

    Args:
        model: PyTorch model
        input_seqs: tensor (N, C, L)
        dinucleotide_shuffle_fn: function generating shuffled baselines
        target_class: output index to attribute
        num_baselines: number of shuffles
        cleanup_interval: cleanup frequency
        device: 'cuda' or 'cpu'

    Returns:
        Tensor (N, C, L): averaged attributions
    """
    model.eval()
    deeplift = DeepLift(model)
    N, C, L = input_seqs.shape

    all_attributions = torch.zeros_like(input_seqs, device="cpu")

    for i, input_seq in enumerate(tqdm(input_seqs, desc="DeepLIFT single-task")):
        input_seq = input_seq.unsqueeze(0).to(device)
        seq_np = input_seq[0].detach().cpu().numpy()

        # Generate baselines
        shuffled_list = [
            torch.tensor(dinucleotide_shuffle_fn(seq_np), dtype=torch.float32)
            for _ in range(num_baselines)
        ]
        baselines = torch.stack(shuffled_list).to(device)
        inputs_repeated = input_seq.repeat(num_baselines, 1, 1)

        # Compute DeepLIFT
        attributions = deeplift.attribute(inputs_repeated, baselines, target=target_class)
        attributions_avg = attributions.mean(dim=0, keepdim=True)

        all_attributions[i] = attributions_avg.squeeze(0).detach().cpu()

        # Cleanup
        del input_seq, baselines, inputs_repeated, attributions, attributions_avg
        if (i + 1) % cleanup_interval == 0 or i == N - 1:
            torch.cuda.empty_cache()

    print("Shape of averaged attributions:", all_attributions.shape)
    return all_attributions


# =========================================================
# 🌐 MULTI-TASK / PLEIOTROPIC DEEPLIFT (Module-Safe)
# =========================================================
def compute_pleiotropic_attributions(
    model,
    input_seqs,
    activity_mask,
    dinucleotide_shuffle_fn,
    method="deeplift",          # "deeplift" or "integrated_gradients"
    num_baselines=10,
    cleanup_interval=50,
    ig_steps=50,                # only used for IG
    device="cuda"
):
    """
    Compute attributions averaged across active tasks (mask=1),
    using either DeepLIFT or Integrated Gradients.

    Args:
        model: multi-output PyTorch model (outputs: batch×T)
        input_seqs: (N, 4, L) tensor
        activity_mask: (N, T) binary mask (1=active, 0=inactive)
        dinucleotide_shuffle_fn: function to generate baseline sequences
        method: 'deeplift' or 'integrated_gradients'
        num_baselines: number of dinuc-shuffled baselines
        cleanup_interval: how often to clear GPU cache
        ig_steps: number of steps for Integrated Gradients
        device: 'cuda' or 'cpu'

    Returns:
        Tensor (N, 4, L): pleiotropic attributions
    """

    # -------------------------------
    # Initialize
    # -------------------------------
    model.eval()
    activity_mask = torch.as_tensor(activity_mask, dtype=torch.float32, device=device)
    N, C, L = input_seqs.shape
    all_attributions = torch.zeros((N, C, L), dtype=torch.float32, device="cpu")

    for i, input_seq in enumerate(tqdm(input_seqs, desc=f"Pleiotropic {method.upper()}")):
        input_seq = input_seq.unsqueeze(0).to(device)
        seq_np = input_seq[0].detach().cpu().numpy()
        task_mask = activity_mask[i]

        # Skip inactive examples
        if task_mask.sum() == 0:
            all_attributions[i] = 0.0
            continue

        # -------------------------------
        # Wrap model for masked mean output
        # -------------------------------
        class MaskedMeanWrapper(nn.Module):
            def __init__(self, base_model, mask):
                super().__init__()
                self.base_model = base_model
                self.register_buffer("mask", mask)

            def forward(self, x):
                y = self.base_model(x)  # shape: (batch, T)
                weighted = y * self.mask
                mean_output = weighted.sum(dim=1, keepdim=True) / self.mask.sum()
                return mean_output  # shape: (batch, 1)

        wrapped_model = MaskedMeanWrapper(model, task_mask)

        # -------------------------------
        # Initialize attribution method
        # -------------------------------
        if method.lower() == "deeplift":
            explainer = DeepLift(wrapped_model)
        elif method.lower() in ["ig", "integrated_gradients"]:
            explainer = IntegratedGradients(wrapped_model)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'deeplift' or 'integrated_gradients'.")

        # -------------------------------
        # Generate baselines
        # -------------------------------
        shuffled_list = [
            torch.tensor(dinucleotide_shuffle_fn(seq_np), dtype=torch.float32)
            for _ in range(num_baselines)
        ]
        baselines = torch.stack(shuffled_list).to(device)
        inputs_repeated = input_seq.repeat(num_baselines, 1, 1)

        # -------------------------------
        # Compute attributions
        # -------------------------------
        if method.lower() == "deeplift":
            attributions = explainer.attribute(inputs_repeated, baselines=baselines)
        else:  # Integrated Gradients
            attributions = explainer.attribute(
                inputs_repeated, baselines=baselines, n_steps=ig_steps
            )

        # -------------------------------
        # Average across baselines
        # -------------------------------
        attributions_avg = attributions.mean(dim=0, keepdim=True)

        # Mask with one-hot to keep only the active base at each position
        attributions_avg = attributions_avg * input_seq

        # Store on CPU
        all_attributions[i] = attributions_avg.squeeze(0).detach().cpu()

        # -------------------------------
        # Cleanup
        # -------------------------------
        del input_seq, baselines, inputs_repeated, attributions, attributions_avg, explainer, wrapped_model
        if (i + 1) % cleanup_interval == 0 or i == N - 1:
            torch.cuda.empty_cache()

    print(f"Shape of {method} pleiotropic attributions:", all_attributions.shape)
    return all_attributions
