import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class Sequence(Dataset):
    
    """
    Dataset class for handling sequences and signals for training a neural network.
    
    Parameters:
        - one_hot_file (str): Path to the file containing one-hot encoded sequences.
        - signal_file (str): Path to the file containing signal data.
        - device (str, optional): Device to load data (default is 'cpu').
        - sqrt_transform (bool, optional): Apply square root transformation to signal data (default is True).

    Methods:
        - __len__(): Get the number of samples in the dataset.
        - __getitem__(index): Get a sample from the dataset by index.

    Notes:
        - The class loads one-hot encoded DNA sequences and signal data for neural network training.
        - It allows customization of the device and signal data transformation.
    """
    
    def __init__(self, one_hot_file: str, 
                 device: str = 'cpu'):
        
        self.one_hot_data = np.load(one_hot_file)
        self.device = device

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        
        return len(self.one_hot_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset by index.

        Parameters:
            - index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing one-hot encoded sequence tensor and signal tensor.
        """
        
        one_hot_tensor = torch.tensor(SequenceSignal.one_hot_encoding(self.one_hot_data[index]), 
                                      dtype = torch.float32, 
                                      device = self.device)

        
        return one_hot_tensor
    
    @staticmethod
    def one_hot_encoding(sequence: str, num_mutations: int = 0) -> np.ndarray:
        """
        Encode a DNA sequence to a one-hot encoded matrix and add N mutations during encoding.

        Parameters:
            - sequence (str): DNA sequence to be encoded.
            - num_mutations (int): Number of random mutations to introduce.

        Returns:
            np.ndarray: One-hot encoded matrix of shape (4 x seq_length).
                    N's are treated as a uniform distribution (0.25 in all 4 channels that sum to 1).
        """
    
        # Define nucleotide mapping and initialization
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_len = len(sequence)
        encoding = np.zeros((4, seq_len), dtype=np.float32)
        sequence_upper = sequence.upper()

        # Convert sequence to integer indices
        indices = np.array([mapping.get(base, 4) for base in sequence_upper])

        # Apply base encoding for A, C, G, T, and N
        valid_indices = indices[indices < 4]  # Exclude 'N' positions
        encoding[valid_indices, np.arange(seq_len)[indices < 4]] = 1.0

        # Handle 'N' positions
        n_positions = (indices == 4)
        encoding[:, n_positions] = 0.25

        # Mutation Handling
        if num_mutations > 0:
            # Choose mutation positions and mutation types
            mutation_positions = np.random.choice(seq_len, size=num_mutations, replace=False)
            mutation_types = np.random.choice(4, size=num_mutations)

            # Apply mutations
            encoding[:, mutation_positions] = 0  # Reset positions to zero
            encoding[mutation_types, mutation_positions] = 1.0  # Apply mutations

        return encoding






class SequenceSignal(Sequence):
    
    """
    Dataset class for handling sequences and signals for training a neural network.
    
    Parameters:
        - one_hot_file (str): Path to the file containing one-hot encoded sequences.
        - signal_file (str): Path to the file containing signal data.
        - device (str, optional): Device to load data (default is 'cpu').
        - sqrt_transform (bool, optional): Apply square root transformation to signal data (default is True).

    Methods:
        - __len__(): Get the number of samples in the dataset.
        - __getitem__(index): Get a sample from the dataset by index.

    Notes:
        - The class loads one-hot encoded DNA sequences and signal data for neural network training.
        - It allows customization of the device and signal data transformation.
    """
    
    def __init__(self, one_hot_file: str, signal_file: str, 
                 device: str = 'cpu', sqrt_transform: bool = True, 
                 num_mutations = 0):
        
        super().__init__(one_hot_file, device = device)
        self.num_mutations = num_mutations
        if sqrt_transform:
            self.signal_data = np.sqrt(np.load(signal_file))
        else:
            self.signal_data = np.load(signal_file)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset by index.

        Parameters:
            - index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing one-hot encoded sequence tensor and signal tensor.
        """
        
        one_hot_tensor = torch.tensor(SequenceSignal.one_hot_encoding(self.one_hot_data[index], 
                                                                      num_mutations = self.num_mutations), 
                                      dtype = torch.float32, 
                                      device = self.device)
        
        signal_tensor = torch.tensor(self.signal_data[index], 
                                     dtype = torch.float32, 
                                     device = self.device)
        
        return one_hot_tensor, signal_tensor


def loadDataset(train_dataset: SequenceSignal, 
                test_dataset: SequenceSignal,
                batch_size: int, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Load train and validation datasets into DataLoader objects.

    Parameters:
        - train_dataset (SequenceSignal): Training dataset.
        - test_dataset (SequenceSignal): Validation dataset.
        - batch_size (int): Batch size for DataLoader.
        - shuffle (bool, optional): Whether to shuffle the data (default is True).

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing DataLoader objects for the training and validation datasets.
    """
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
    )

    return (train_loader, valid_loader)


