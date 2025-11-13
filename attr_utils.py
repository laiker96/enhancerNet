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
    """

    model.eval()
    activity_mask = torch.as_tensor(activity_mask, dtype=torch.float32, device=device)
    N, C, L = input_seqs.shape
    all_attributions = torch.zeros((N, C, L), dtype=torch.float32, device="cpu")

    for i, input_seq in enumerate(tqdm(input_seqs, desc=f"Pleiotropic {method.upper()}")):
        input_seq = input_seq.unsqueeze(0).to(device)
        seq_np = input_seq[0].detach().cpu().numpy()
        task_mask = activity_mask[i]

        if task_mask.sum() == 0:
            all_attributions[i] = 0.0
            continue

        class MaskedMeanWrapper(nn.Module):
            def __init__(self, base_model, mask):
                super().__init__()
                self.base_model = base_model
                self.register_buffer("mask", mask)

            def forward(self, x):
                y = self.base_model(x)
                weighted = y * self.mask
                mean_output = weighted.sum(dim=1, keepdim=True) / self.mask.sum()
                return mean_output

        wrapped_model = MaskedMeanWrapper(model, task_mask)

        if method.lower() == "deeplift":
            explainer = DeepLift(wrapped_model)
        elif method.lower() in ["ig", "integrated_gradients"]:
            explainer = IntegratedGradients(wrapped_model)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'deeplift' or 'integrated_gradients'.")

        shuffled_list = [
            torch.tensor(dinucleotide_shuffle_fn(seq_np), dtype=torch.float32)
            for _ in range(num_baselines)
        ]
        baselines = torch.stack(shuffled_list).to(device)
        inputs_repeated = input_seq.repeat(num_baselines, 1, 1)

        if method.lower() == "deeplift":
            attributions = explainer.attribute(inputs_repeated, baselines=baselines)
        else:
            attributions = explainer.attribute(
                inputs_repeated, baselines=baselines, n_steps=ig_steps
            )

        attributions_avg = attributions.mean(dim=0, keepdim=True)
        attributions_avg = attributions_avg * input_seq
        all_attributions[i] = attributions_avg.squeeze(0).detach().cpu()

        del input_seq, baselines, inputs_repeated, attributions, attributions_avg, explainer, wrapped_model
        if (i + 1) % cleanup_interval == 0 or i == N - 1:
            torch.cuda.empty_cache()

    print(f"Shape of {method} pleiotropic attributions:", all_attributions.shape)
    return all_attributions


# =========================================================
# 🚀 MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from torch.utils.data import DataLoader
    from model_structure.SequenceSignal import Sequence
    from model_structure import transformer_model

    parser = argparse.ArgumentParser(description="Compute DeepLIFT/IG attributions for enhancer models.")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--input_seqs", type=str, required=True, help="Path to .npy or .pt sequence file.")
    parser.add_argument("--activity_mask", type=str, required=True, help="Path to activity mask .txt file.")
    parser.add_argument("--method", type=str, default="deeplift", choices=["deeplift", "ig", "integrated_gradients"])
    parser.add_argument("--num_baselines", type=int, default=10)
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seq_length", type=int, default=1000)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model = transformer_model.TransformerCNNMixtureModel(
        n_conv_layers=4,
        n_filters=[256, 60, 60, 120],
        kernel_sizes=[7, 3, 5, 3],
        dilation=[1, 1, 1, 1],
        drop_conv=0.1,
        n_fc_layers=2,
        drop_fc=0.4,
        n_neurons=[256, 256],
        output_size=9,
        drop_transformer=0.2,
        input_size=4,
        n_encoder_layers=2,
        n_heads=8,
        n_transformer_FC_layers=256,
    ).to(device)

    state = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["network"])
    model.eval()

    # --- Load sequences ---
    dataset = Sequence(args.input_seqs, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    total = len(dataset)
    input_seqs = torch.zeros((total, 4, args.seq_length), device=device)

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            start, end = i * args.batch_size, i * args.batch_size + batch.shape[0]
            input_seqs[start:end] = batch

    # Save input tensor
    np.save(f"{args.output_prefix}_input_seqs.npy", input_seqs.cpu().numpy())
    print(f"Saved input one-hot sequences to {args.output_prefix}_input_seqs.npy")

    # --- Load activity mask ---
    activity_mask = np.loadtxt(args.activity_mask)

    # --- Compute attributions ---
    print(f"Computing {args.method.upper()} with {args.num_baselines} baselines...")
    if args.method.lower() in ["ig", "integrated_gradients"]:
        attributions = compute_pleiotropic_attributions(
            model=model,
            input_seqs=input_seqs,
            activity_mask=activity_mask,
            dinucleotide_shuffle_fn=dinucleotide_shuffle,
            method=args.method,
            num_baselines=args.num_baselines,
            ig_steps=args.ig_steps,
            device=device,
        )
    else:
        attributions = compute_pleiotropic_attributions(
            model=model,
            input_seqs=input_seqs,
            activity_mask=activity_mask,
            dinucleotide_shuffle_fn=dinucleotide_shuffle,
            method=args.method,
            num_baselines=args.num_baselines,
            device=device,
        )

    np.save(f"{args.output_prefix}_{args.method}_attributions.npy", attributions.cpu().numpy())
    print(f"✅ Saved {args.method} attributions to {args.output_prefix}_{args.method}_attributions.npy")