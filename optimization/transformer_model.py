import torch
from torch import nn
from .NNBlocks import (SqueezeExciteBlock, 
                       ConvPoolingBlock, 
                       PositionalEncoding, 
                       ConvBlock, LinearBlock)

class TransformerCNNMixtureModel(nn.Module):
    
    """
    Transformer CNN hybrid model class for optuna optimization
    """
    
    def __init__(self, n_conv_layers, 
                 n_filters, kernel_sizes, 
                 dilation, drop_conv, 
                 n_encoder_layers, n_heads, 
                 n_transformer_FC_layers, 
                 drop_transformer, 
                 n_fc_layers, n_neurons, 
                 drop_fc, 
                 input_size = 4, 
                 output_size = 32, 
                 sequence_length = 1024):
        """
        Parameters:
            
        Convolutional layers hyperparameters
        ---------------------------------------------
            - n_conv_layers (int):               Number of convolutional layers
            - n_filters (list):                  Number of filters of conv layers
            - kernel_sizes (list):               Size of kernels of conv layers
            - dilation (list):                   Dilation factors for conv layers
            - drop_conv (list):                  Dropout rates for conv layers
            
        Transformer encoder layers hyperparameters
        ---------------------------------------------
            - n_encoder_layers (int):            Number of transformerEncoder layers
            - n_heads (int):                     Number of multiattention heads
            - n_transformer_FC_layers (int):     Dimension of the FC layers of transformerEncoder layers
            - drop_transformer (float):          Dropout rate for the transformerEncoder layer
            
        Fully Connected layers hyperparameters
        ---------------------------------------------
            - n_fc_layers (int):                 Number of FC layers
            - n_neurons (list):                  Number of neurons of FC layers
            - drop_fc1 (float):                  Dropout ratio for FC layers
            - input_size (int):                  Number of input channels (4 for DNA sequences)
            - output_size (int):                 Number of output neurons
            - sequence_length (int):             Input DNA sequence length
        """
        
        super().__init__()
        # Define the convolutional layers of the network
        conv_block_0 = nn.Sequential(
            ConvPoolingBlock(input_size, n_filters[0], 
                             kernel_size = kernel_sizes[0], 
                             dilation = dilation[0], 
                             dropout = drop_conv))
        
        self.convs = nn.ModuleList([conv_block_0])
        # Calculation of sequence length after conv layer + max_pooling with size 2
        out_size = sequence_length - dilation[0] * (kernel_sizes[0] - 1)
        out_size = int(out_size / 2)
        
        for i in range(1, n_conv_layers):
            
            conv_block = nn.Sequential(
                ConvPoolingBlock(n_filters[i-1], n_filters[i], 
                                 kernel_size = kernel_sizes[i], 
                                 dilation = dilation[i], 
                                 dropout = drop_conv))
            
            self.convs.append(conv_block)
            
            out_size = out_size - dilation[i] * (kernel_sizes[i] - 1)
            out_size = int(out_size/2)
        
        # Positional encoding layer
        d_model = n_filters[-1] # Embedding dimension of the attention layers
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Define the transformer encoder layers 
        transformerlayer = torch.nn.TransformerEncoderLayer(d_model, n_heads, 
                                                            dim_feedforward = n_transformer_FC_layers, 
                                                            dropout = drop_transformer, 
                                                            batch_first = True)
        
        self.transformerEncoder = torch.nn.TransformerEncoder(transformerlayer, 
                                                              num_layers = n_encoder_layers)
        # Calculate size of first FC layer
        self.out_feature = n_filters[-1]
        
        self.fcs = nn.ModuleList([LinearBlock(self.out_feature*2, n_neurons[0], dropout = drop_fc)])
        #self.fcs = nn.ModuleList([LinearBlock(self.out_feature, n_neurons[0], dropout = drop_fc)])
        
        for j in range(1, n_fc_layers):
            self.fcs.append(LinearBlock(n_neurons[j-1], n_neurons[j], dropout = drop_fc))
        
        # Define the output layer
        self.output_layer = nn.Linear(n_neurons[-1], output_size)
               
    def forward(self, x):
        
        """
        Forward propagation.
        
        Parameters:
            - x (torch.Tensor): Input tensor of size [N,input_size,sequence_length]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,output_size]

        """
        
        for i, conv_i in enumerate(self.convs):
            x = conv_i(x)
        
        # Change input tensor shape for the multiheadattention layers
        # from [batch_size, features, seq_length] to [batch_size, seq_length, features]
        x = self.positional_encoding(x.swapaxes(1, 2))
        x = self.transformerEncoder(x)
        
        # Reshape for FC layers
        #x = x.view(-1, self.out_feature)
        x_max, _ = torch.max(x, dim = 1)
        x_mean = torch.mean(x, dim = 1)
        #x = torch.mean(x, dim = 1)
        x = torch.cat((x_max, x_mean), dim = 1)
        
        for j, fc_j in enumerate(self.fcs):
            x = fc_j(x)
        x = self.output_layer(x)
        
        return x
