import torch
from torch import nn
from typing import List
from .NNBlocks import (ConvPoolingBlock, 
                       PositionalEncoding, 
                       LinearBlock)

class TransformerCNNMixtureModel(nn.Module):
    """
    Hybrid Transformer-CNN model for processing sequential data (e.g., DNA sequences).
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
                 output_size: int = 9,
                 sequence_length: int = 1000) -> None:
        """
        Initialize the TransformerCNNMixtureModel.

        Args:
            n_conv_layers (int): Number of convolutional layers.
            n_filters (List[int]): Number of filters in each convolutional layer.
            kernel_sizes (List[int]): Kernel sizes for each convolutional layer.
            dilation (List[int]): Dilation factors for each convolutional layer.
            drop_conv (float): Dropout rate for all convolutional layers.

            n_encoder_layers (int): Number of TransformerEncoder layers.
            n_heads (int): Number of attention heads in each encoder layer.
            n_transformer_FC_layers (int): Hidden size of feedforward layers in Transformer.
            drop_transformer (float): Dropout rate in the transformer layers.

            n_fc_layers (int): Number of fully connected layers.
            n_neurons (List[int]): Number of neurons in each fully connected layer.
            drop_fc (float): Dropout rate for fully connected layers.

            input_size (int): Number of input channels (e.g., 4 for DNA sequences).
            output_size (int): Number of output neurons.
            sequence_length (int): Length of the input sequence.
        """
        super().__init__()

        # Define the convolutional layers of the network
        conv_block_0 = nn.Sequential(
            ConvPoolingBlock(input_size, n_filters[0], 
                             kernel_size=kernel_sizes[0], 
                             dilation=dilation[0], 
                             dropout=drop_conv))
        
        self.convs = nn.ModuleList([conv_block_0])

        # Calculation of sequence length after conv layer + max_pooling with size 2
        out_size = sequence_length - dilation[0] * (kernel_sizes[0] - 1)
        out_size = int(out_size / 2)
        
        for i in range(1, n_conv_layers):
            conv_block = nn.Sequential(
                ConvPoolingBlock(n_filters[i - 1], n_filters[i], 
                                 kernel_size=kernel_sizes[i], 
                                 dilation=dilation[i], 
                                 dropout=drop_conv))
            self.convs.append(conv_block)
            
            out_size = out_size - dilation[i] * (kernel_sizes[i] - 1)
            out_size = int(out_size / 2)
        
        # Positional encoding layer
        d_model = n_filters[-1]  # Embedding dimension of the attention layers
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Define the transformer encoder layers 
        transformerlayer = torch.nn.TransformerEncoderLayer(
            d_model, n_heads, 
            dim_feedforward=n_transformer_FC_layers, 
            dropout=drop_transformer, 
            batch_first=True)
        
        self.transformerEncoder = torch.nn.TransformerEncoder(
            transformerlayer, num_layers=n_encoder_layers)
        
        # Calculate size of first FC layer
        self.out_feature = n_filters[-1]
        self.fcs = nn.ModuleList([
            LinearBlock(self.out_feature * 2, n_neurons[0], dropout=drop_fc)
        ])
        
        for j in range(1, n_fc_layers):
            self.fcs.append(LinearBlock(n_neurons[j - 1], n_neurons[j], dropout=drop_fc))
        
        # Define the output layer
        self.output_layer = nn.Linear(n_neurons[-1], output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size, sequence_length]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        for i, conv_i in enumerate(self.convs):
            x = conv_i(x)
        
        # Reshape for transformer: [batch_size, sequence_length, features]
        x = self.positional_encoding(x.swapaxes(1, 2))
        x = self.transformerEncoder(x)
        
        # Pooling across sequence dimension
        x_max, _ = torch.max(x, dim=1)
        x_mean = torch.mean(x, dim=1)
        x = torch.cat((x_max, x_mean), dim=1)
        
        for j, fc_j in enumerate(self.fcs):
            x = fc_j(x)
        
        x = self.output_layer(x)
        return x
