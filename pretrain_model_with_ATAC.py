#!/usr/bin/env python
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from model_structure import SequenceSignal, transformer_model, train_val_loops


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train CNN+Transformer on ATAC-seq data with optional Z-score normalization"
    )
    parser.add_argument("--x_train", type=str, required=True, help="Path to training features (npy)")
    parser.add_argument("--y_train", type=str, required=True, help="Path to training labels (npy)")
    parser.add_argument("--x_val", type=str, required=True, help="Path to validation features (npy)")
    parser.add_argument("--y_val", type=str, required=True, help="Path to validation labels (npy)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output_shape", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_path", type=str, default="ATAC_transformer.pth")
    parser.add_argument("--z_mean", type=str, default=None, help="Optional path to Z-score means (npy)")
    parser.add_argument("--z_std", type=str, default=None, help="Optional path to Z-score stds (npy)")
    parser.add_argument("--max_lr", type=float, default=2e-3)
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile() for performance")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load datasets ---
    N_TRAIN_EXAMPLES = np.load(args.y_train).shape[0]
    dataloaders = SequenceSignal.load_dataset(
        Path(args.x_train),
        Path(args.y_train),
        Path(args.x_val),
        Path(args.y_val),
        args.batch_size,
        device=device
    )

    # --- Model definition ---
    model = transformer_model.TransformerCNNMixtureModel(
        n_conv_layers=4,
        n_filters=[256, 60, 60, 120],
        kernel_sizes=[7, 3, 5, 3],
        dilation=[1, 1, 1, 1],
        drop_conv=0.1,
        n_fc_layers=2,
        drop_fc=0.4,
        n_neurons=[256, 256],
        output_size=args.output_shape,
        drop_transformer=0.3,
        #drop_transformer=0.2,
        input_size=4,
        #n_encoder_layers=2,
        n_encoder_layers=3,
        n_heads=8,
        n_transformer_FC_layers=256).to(device)

    # --- Optionally compile model (PyTorch 2.0+) ---
    if args.compile:
        print("Compiling model with torch.compile() for optimized performance...")
        model = torch.compile(model)  # speeds up training in PyTorch 2.x

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=args.n_epochs,
        max_lr=args.max_lr,
        steps_per_epoch=math.ceil(N_TRAIN_EXAMPLES / args.batch_size),
        pct_start=0.15,
        anneal_strategy="linear"
    )
    criterion = nn.MSELoss()

    # --- Optional normalization paths ---
    means_path = Path(args.z_mean) if args.z_mean else None
    stds_path = Path(args.z_std) if args.z_std else None

    # --- Checkpoint loading ---
    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.exists():
        training_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(training_state["network"])
        optimizer.load_state_dict(training_state["optimizer"])
        lr_scheduler.load_state_dict(training_state["lr_sched"])
        best_valid_loss = training_state.get("best_valid_loss", None)
        print(f"Resumed training. Best validation loss so far: {best_valid_loss}")

    # --- Train ---
    train_val_loops.train_N_epochs(
        model,
        optimizer,
        criterion=criterion,
        train_loader=dataloaders[0],
        valid_loader=dataloaders[1],
        num_epochs=args.n_epochs,
        patience=args.patience,
        model_path=checkpoint_path,
        lr_scheduler=lr_scheduler,
        means_path=means_path,
        stds_path=stds_path,
        DEVICE=device,
        use_amp=args.use_amp
    )


if __name__ == "__main__":
    main()
