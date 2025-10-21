#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from model_structure import transformer_model, SequenceSignal

def main():
    parser = argparse.ArgumentParser(description="FASTA -> one-hot -> model predictions -> npy")
    parser.add_argument("fasta_file", type=str, help="Input FASTA file")
    parser.add_argument("output_file", type=str, help="Output .npy file with probabilities")
    parser.add_argument("--checkpoint", type=str, default="model_weights/pretrained_model.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Step 1: One-hot encode the sequences
    os.system(f"python encode_fasta.py --fasta {args.fasta_file} --output temp_input.npy")

    # Step 2: Load encoded sequences
    X = np.load("temp_input.npy")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Step 3: Load the model
    model = transformer_model.TransformerCNNMixtureModel(
        n_conv_layers=4,
        n_filters=[256, 60, 60, 120],
        kernel_sizes=[7, 3, 5, 3],
        dilation=[1, 1, 1, 1],
        drop_conv=0.1,
        n_fc_layers=2,
        drop_fc=0.4,
        n_neurons=[256, 256],
        output_size=9,  # number of independent tasks
        drop_transformer=0.2,
        input_size=4,
        n_encoder_layers=2,
        n_heads=8,
        n_transformer_FC_layers=256
    )

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["network"])
    model.to(device)
    model.eval()

    # Step 4: Predict with sigmoid per task (independent probabilities)
    all_preds = []
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)  # independent probability per task
            all_preds.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    np.save(args.output_file, y_pred)
    print(f"Predictions saved to {args.output_file}, shape: {y_pred.shape}")

if __name__ == "__main__":
    main()

