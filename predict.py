#!/usr/bin/env python
import argparse
import numpy as np
import torch
from pathlib import Path
from optimization import transformer_model

def main():
    parser = argparse.ArgumentParser(description="Predict enhancer scores with trained CNN+Transformer model")
    parser.add_argument("--input_npy", type=str, required=True, help="Input one-hot encoded sequences (npy)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output npy file for predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X = np.load(args.input_npy)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    output_shape = X_tensor.shape[-1]  # optional; depends on training
    model = transformer_model.TransformerCNNMixtureModel(
        n_conv_layers=4,
        n_filters=[256, 60, 60, 120],
        kernel_sizes=[7,3,5,3],
        dilation=[1,1,1,1],
        drop_conv=0.1,
        n_fc_layers=2,
        drop_fc=0.4,
        n_neurons=[256,256],
        output_size=None,  # will infer from checkpoint
        drop_transformer=0.2,
        input_size=4,
        n_encoder_layers=2,
        n_heads=8,
        n_transformer_FC_layers=256
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['network'])
    model.to(device)
    model.eval()

    # Predict
    all_preds = []
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    np.save(args.output, y_pred)
    print(f"Predictions saved to {args.output}, shape: {y_pred.shape}")

if __name__ == "__main__":
    main()