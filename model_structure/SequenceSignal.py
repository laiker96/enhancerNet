import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class Sequence(Dataset):
    """
    Efficient dataset class for handling large one-hot encoded DNA sequences.

    Parameters:
        one_hot_file (str): Path to the .npy file containing one-hot encoded sequences.
        device (torch.device, optional): Target device for on-demand transfer (default: CPU).

    Notes:
        - Uses memory mapping to avoid loading the entire dataset into memory.
        - Transfers sequences to GPU only when fetched.
    """
    
    def __init__(self, 
                 one_hot_file: str, 
                 device: torch.device = torch.device('cpu')):

        self.one_hot_data = np.load(one_hot_file, mmap_mode='r')
        self.device = device

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.one_hot_data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Fetch one sequence, transfer to device on demand.

        Returns:
            torch.Tensor: One-hot encoded sequence tensor.
        """

        seq = torch.from_numpy(np.array(self.one_hot_data[index], dtype=np.float32))


        if self.device.type == 'cuda':
            seq = seq.to(self.device, non_blocking=True)

        return seq


class SequenceSignal(Sequence):
    """
    Dataset class for handling one-hot encoded sequences and corresponding signal data.

    Parameters:
        one_hot_file (str): Path to .npy file containing one-hot encoded sequences.
        signal_file (str): Path to .npy file containing signal data.
        device (torch.device, optional): Target device for on-demand transfer (default: CPU).

    Notes:
        - Uses memory mapping for both datasets to handle large files efficiently.
        - Transfers only the current sample to GPU when fetched.
    """

    def __init__(
        self, 
        one_hot_file: str, 
        signal_file: str, 
        device: torch.device = torch.device('cpu')
    ):
        super().__init__(one_hot_file, device=device)
        self.signal_data = np.load(signal_file, mmap_mode='r')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a (sequence, signal) pair as tensors."""
        # Sequence tensor

        seq = torch.from_numpy(np.array(self.one_hot_data[index], dtype=np.float32))
        signal = torch.from_numpy(np.array(self.signal_data[index], dtype=np.float32))
        return seq, signal

def _loadDataset(
    train_dataset: Dataset, 
    val_dataset: Dataset,
    batch_size: int, 
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and validation datasets.

    Parameters:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool, optional): Whether to shuffle the training data (default: True).
        num_workers (int, optional): Number of subprocesses for data loading (default: 4).
        pin_memory (bool, optional): Whether to pin memory for faster GPU transfer (default: True).

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def load_dataset(
    train_encoding_path: str, 
    train_signal_path: str, 
    val_encoding_path: str, 
    val_signal_path: str, 
    batch_size: int, 
    device: torch.device = torch.device('cpu'), 
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare the training and validation datasets.

    Args:
        train_encoding_path (str): Path to the training encoding (.npy) file.
        train_signal_path (str): Path to the training signal (.npy) file.
        val_encoding_path (str): Path to the validation encoding (.npy) file.
        val_signal_path (str): Path to the validation signal (.npy) file.
        batch_size (int): Batch size for training.
        device (torch.device): Target device ('cpu' or 'cuda').
        shuffle (bool): Whether to shuffle the training data.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to pin memory for faster transfer to GPU.
        **kwargs: Additional keyword arguments for SequenceSignal.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """

    train_dataset = SequenceSignal(
        train_encoding_path, 
        train_signal_path, 
        device=device, 
        **kwargs
    )

    val_dataset = SequenceSignal(
        val_encoding_path, 
        val_signal_path, 
        device=device, 
        **kwargs
    )

    dataloaders = _loadDataset(
        train_dataset, 
        val_dataset, 
        batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloaders

