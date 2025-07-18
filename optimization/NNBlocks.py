import torch
from torch import nn
import math

class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.4
    ) -> None:
        """
        Initialize a ConvPoolingBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_rate (float, optional): Dropout probability. Default is 0.4.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout),
            nn.ReLU()
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvPoolingBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, reduced_length].
        """
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.4
    ) -> None:
        """
        Initialize a ConvPoolingBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int, optional): Dilation factor for convolution. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.4.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Dropout1d(dropout),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvPoolingBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, reduced_length].
        """
        return self.block(x)


class ConvPoolingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.4
    ) -> None:
        """
        Initialize a ConvPoolingBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int, optional): Dilation factor for convolution. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.4.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(2, 2)
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvPoolingBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, reduced_length].
        """
        return self.block(x)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, c: int, r: int = 16) -> None:
        """
        Initialize a SqueezeExciteBlock.

        Args:
            c (int): Number of input channels.
            r (int, optional): Reduction factor. Default is 16.
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SqueezeExciteBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, c, sequence_length].

        Returns:
            torch.Tensor: Output tensor with channel-wise scaling applied.
        """
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initialize a PositionalEncoding layer.

        Args:
            d_model (int): Dimension of the model embeddings.
            max_len (int, optional): Maximum sequence length. Default is 5000.
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, embedding_dim].

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(1)].swapaxes(0, 1)
        return x
