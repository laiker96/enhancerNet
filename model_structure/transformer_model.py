import torch
from torch import nn
from typing import List
from .NNBlocks import ConvPoolingBlock, PositionalEncoding, LinearBlock


class TransformerCNNMixtureModel(nn.Module):
    """
    Hybrid CNN–Transformer model for sequence data (e.g., DNA, ATAC-seq, etc.).

    This architecture combines local feature extraction via convolutional blocks
    with long-range dependency modeling via Transformer encoder layers. After the
    transformer stack, the model performs global pooling (max + mean) and passes
    the concatenated features through fully connected layers for final prediction.

    Args:
        n_conv_layers (int): 
            Number of convolutional layers.

        n_filters (List[int]): 
            List of filter counts for each convolutional layer. 
            Length must equal `n_conv_layers`.

        kernel_sizes (List[int]): 
            List of kernel sizes for each convolutional layer.
            Length must equal `n_conv_layers`.

        dilation (List[int]): 
            List of dilation factors for each convolutional layer.
            Length must equal `n_conv_layers`.

        drop_conv (float): 
            Dropout rate applied after each convolutional block.

        n_encoder_layers (int): 
            Number of Transformer encoder layers to stack.

        n_heads (int): 
            Number of attention heads in each Transformer encoder layer.

        n_transformer_FC_layers (int): 
            Hidden size (feed-forward dimension) inside each Transformer encoder layer.

        drop_transformer (float): 
            Dropout rate inside each Transformer encoder layer (applied to attention
            and feedforward sublayers).

        n_fc_layers (int): 
            Number of fully connected (dense) layers after Transformer pooling.

        n_neurons (List[int]): 
            List of neuron counts for each fully connected layer.
            Length must equal `n_fc_layers`.

        drop_fc (float): 
            Dropout rate applied after each fully connected layer.

        input_size (int, optional): 
            Number of input channels (default = 4, e.g., one-hot encoded DNA bases).

        output_size (int, optional): 
            Number of output neurons (default = 9, e.g., regression targets or classes).
    """

    def __init__(self,
                 n_conv_layers: int,
                 n_filters: List[int],
                 kernel_sizes: List[int],
                 dilation: List[int],
                 drop_conv: float,
                 n_encoder_layers: int,
                 n_heads: int,
                 n_transformer_FC_layers: int,
                 drop_transformer: float,
                 n_fc_layers: int,
                 n_neurons: List[int],
                 drop_fc: float,
                 input_size: int = 4,
                 output_size: int = 9) -> None:
        super().__init__()

        # === Convolutional feature extractor ===
        conv_block_0 = nn.Sequential(
            ConvPoolingBlock(
                input_size, n_filters[0],
                kernel_size=kernel_sizes[0],
                dilation=dilation[0],
                dropout=drop_conv
            )
        )
        self.convs = nn.ModuleList([conv_block_0])

        for i in range(1, n_conv_layers):
            conv_block = nn.Sequential(
                ConvPoolingBlock(
                    n_filters[i - 1], n_filters[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilation[i],
                    dropout=drop_conv
                )
            )
            self.convs.append(conv_block)

        # === Positional encoding ===
        d_model = n_filters[-1]
        self.positional_encoding = PositionalEncoding(d_model)

        # === Independent Transformer encoder layers (with GELU) ===
        self.transformerEncoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=n_transformer_FC_layers,
                dropout=drop_transformer,
                activation='gelu',
                batch_first=True
            )
            for _ in range(n_encoder_layers)
        ])

        # === Fully connected classifier ===
        self.out_feature = n_filters[-1]
        self.fcs = nn.ModuleList([
            LinearBlock(self.out_feature * 2, n_neurons[0], dropout=drop_fc)
        ])
        for j in range(1, n_fc_layers):
            self.fcs.append(
                LinearBlock(n_neurons[j - 1], n_neurons[j], dropout=drop_fc)
            )

        self.output_layer = nn.Linear(n_neurons[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): 
                Input tensor of shape `[batch_size, input_size, sequence_length]`.

        Returns:
            torch.Tensor: 
                Output tensor of shape `[batch_size, output_size]`.
        """
        # === CNN feature extraction ===
        for conv_i in self.convs:
            x = conv_i(x)

        # === Transformer encoder stack ===
        x = self.positional_encoding(x.swapaxes(1, 2))
        for layer in self.transformerEncoder:
            x = layer(x)

        # === Global pooling (max + mean) ===
        x_max, _ = torch.max(x, dim=1)
        x_mean = torch.mean(x, dim=1)
        x = torch.cat((x_max, x_mean), dim=1)

        # === Fully connected layers ===
        for fc_j in self.fcs:
            x = fc_j(x)

        # === Output layer ===
        return self.output_layer(x)

