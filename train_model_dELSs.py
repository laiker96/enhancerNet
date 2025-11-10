#!/usr/bin/env python
"""
Fine-tune a pretrained TransformerCNNMixtureModel for CRE enhancer classification.
Supports optional mixed precision, model compilation (PyTorch 2.x), checkpointing,
and optional OneCycleLR warmup scheduling.
"""

import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from model_structure import SequenceSignal, transformer_model, train_val_loops


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune CNN+Transformer for CRE enhancer classification")

    # --- Data ---
    parser.add_argument("--x_train", type=str, required=True, help="Training features (.npy)")
    parser.add_argument("--y_train", type=str, required=True, help="Training labels (.npy)")
    parser.add_argument("--x_val", type=str, required=True, help="Validation features (.npy)")
    parser.add_argument("--y_val", type=str, required=True, help="Validation labels (.npy)")

    # --- Training ---
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output_shape", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_path", type=str, default="best_model_dELSs.pth")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Optional pretrained weights path")

    # --- Optimization ---
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["sgd", "adam", "adamw"],
                        help="Optimizer type (sgd, adam, or adamw)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--max_lr", type=float, default=2e-3, help="Peak learning rate for OneCycleLR (if enabled)")
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay for optimizer")
    parser.add_argument("--use_lr_scheduler", action="store_true",
                        help="Use OneCycleLR scheduler with warmup")

    # --- Other features ---
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model with torch.compile() for speed (PyTorch 2.x)")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load datasets ---
    N_TRAIN = np.load(args.y_train).shape[0]
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
        drop_transformer=0.2,
        input_size=4,
        n_encoder_layers=2,
        n_heads=8,
        n_transformer_FC_layers=256
    ).to(device)

    # --- Optimizer ---
    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        print(f"🧠 Using SGD optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"🧠 Using Adam optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"🧠 Using AdamW optimizer (lr={args.lr}, weight_decay={args.weight_decay})")

    # --- Scheduler (optional OneCycleLR) ---
    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=args.n_epochs,
            max_lr=args.max_lr,
            steps_per_epoch=math.ceil(N_TRAIN / args.batch_size),
            pct_start=0.15,  # fixed warmup phase
            anneal_strategy="linear"
        )
        print(f"✅ Using OneCycleLR scheduler with max_lr={args.max_lr}, warmup=15%")
    else:
        print("⚠️  Training without learning rate scheduler (fixed LR).")

    # --- BCE loss ---
    epsilon = 1e-6
    y_train_array = np.load(args.y_train)
    n_pos = y_train_array.sum(axis=0)
    n_neg = (1.0 - y_train_array).sum(axis=0)
    pos_weight = torch.log1p(torch.tensor(n_neg / (n_pos + epsilon))).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)  # can set to pos_weight if needed

    checkpoint_path = Path(args.checkpoint_path)
    best_valid_loss = float("inf")

    # --- Load pretrained weights ---
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        state = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(state["network"], strict=False)
        print("✅ Pretrained weights loaded successfully.")

    # --- Resume training if checkpoint exists ---
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["network"], strict=False)

        # --- Check optimizer type before loading ---
        old_opt_state = state.get("optimizer", None)
        if old_opt_state is not None:
            # Try to infer optimizer type from the checkpoint
            saved_opt_type = state.get("optimizer_type", None)
            current_opt_type = optimizer.__class__.__name__

            if saved_opt_type == current_opt_type:
                optimizer.load_state_dict(old_opt_state)
                print(f"✅ Optimizer ({current_opt_type}) state restored from checkpoint.")
            else:
                print(f"⚠️ Optimizer type mismatch: checkpoint has {saved_opt_type}, current is {current_opt_type}. Reinitializing optimizer.")
        else:
            print("ℹ️ No optimizer state found in checkpoint, starting fresh.")

        # --- Scheduler restore (only if compatible) ---
        if "lr_sched" in state and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(state["lr_sched"])
                print("✅ Scheduler state restored.")
            except Exception as e:
                print(f"⚠️ Scheduler could not be restored, restarting schedule.\n{e}")

        best_valid_loss = state.get("best_valid_loss", best_valid_loss)
        print(f"Resumed training. Best validation loss so far: {best_valid_loss:.6f}")



    # --- Compile AFTER loading weights ---
    if args.compile:
        print("🚀 Compiling model with torch.compile() for optimized performance...")
        model = torch.compile(model)

    # --- Train ---
    train_val_loops.train_N_epochs(
        network=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=dataloaders[0],
        valid_loader=dataloaders[1],
        num_epochs=args.n_epochs,
        patience=args.patience,
        model_path=checkpoint_path,
        best_valid_loss=best_valid_loss,
        lr_scheduler=lr_scheduler,
        DEVICE=device,
        use_amp=args.use_amp
    )


if __name__ == "__main__":
    main()
