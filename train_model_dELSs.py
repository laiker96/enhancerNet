#!/usr/bin/env python
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from model_structure import (SequenceSignal, 
                             transformer_model, 
                             train_val_loops)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN+Transformer for CRE enhancer classification")
    parser.add_argument("--x_train", type=str, required=True, help="Training features (npy)")
    parser.add_argument("--y_train", type=str, required=True, help="Training labels (npy)")
    parser.add_argument("--x_val", type=str, required=True, help="Validation features (npy)")
    parser.add_argument("--y_val", type=str, required=True, help="Validation labels (npy)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output_shape", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_path", type=str, default="best_model_dELSs.pth")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Optional pretrained weights path")
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--max_lr", type=float, default=2e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load datasets
    N_TRAIN = np.load(args.y_train).shape[0]
    dataloaders = SequenceSignal.load_dataset(
        Path(args.x_train),
        Path(args.y_train),
        Path(args.x_val),
        Path(args.y_val),
        args.batch_size,
        device=device
    )

    # Model
    model = transformer_model.TransformerCNNMixtureModel(
        n_conv_layers=4,
        n_filters=[256, 60, 60, 120],
        kernel_sizes=[7,3,5,3],
        dilation=[1,1,1,1],
        drop_conv=0.1,
        n_fc_layers=2,
        drop_fc=0.4,
        n_neurons=[256,256],
        output_size=args.output_shape,
        drop_transformer=0.2,
        input_size=4,
        n_encoder_layers=2,
        n_heads=8,
        n_transformer_FC_layers=256
    )
    model.to(device)
    summary(model)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=args.n_epochs,
        max_lr=args.max_lr,
        steps_per_epoch=math.ceil(N_TRAIN / args.batch_size),
        pct_start=0.15,
        anneal_strategy="linear"
    )

    # BCE Loss
    epsilon = 1e-6
    y_train_array = np.load(args.y_train)
    n_pos = y_train_array.sum(axis=0)
    n_neg = (1.0 - y_train_array).sum(axis=0)
    pos_weight = torch.log1p(torch.tensor(n_neg / (n_pos + epsilon))).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)  # Enable pos_weight if needed

    checkpoint_path = Path(args.checkpoint_path)

    # Pretrained weights
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        state = torch.load(args.pretrained_weights, weights_only=True, map_location=device)
        model.load_state_dict(state['network'])

    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint {checkpoint_path}")
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state['network'])
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_sched'])
        best_valid_loss = state.get('best_valid_loss', None)
        print(f"Resumed training. Best validation loss so far: {best_valid_loss}")

    # Start/continue training
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
        DEVICE=device,
        use_amp=args.use_amp
    )

if __name__ == "__main__":
    main()
