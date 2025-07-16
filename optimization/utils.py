import pickle
from .SequenceSignal import SequenceSignal, loadDataset
from typing import List, Tuple
from torch.utils.data import DataLoader

def load_dataset(train_encoding_path: str, 
                 train_signal_path: str, 
                 val_encoding_path: str, 
                 val_signal_path: str, 
                 batch_size: int, 
                 device: str, shuffle = True,
                 **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare the training and validation datasets.

    Args:
        train_encoding_path (str): Path to the training encoding file.
        train_signal_path (str): Path to the training signal file.
        val_encoding_path (str): Path to the validation encoding file.
        val_signal_path (str): Path to the validation signal file.
        batch_size (int): Batch size for training.
        device (str): Device to load data ('cpu' or 'cuda').

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing train and validation dataloaders.
    """
    train_dataset = SequenceSignal(train_encoding_path, train_signal_path,
                                                 device=device, **kwargs)

    val_dataset = SequenceSignal(val_encoding_path, val_signal_path, 
                                                device=device, **kwargs)

    dataloaders = loadDataset(train_dataset, val_dataset, batch_size, shuffle = shuffle)
    
    return dataloaders

