import torch
import numpy as np
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional
import torch


def train(network: Module, 
          optimizer: Optimizer, 
          criterion: _Loss, 
          train_loader: Optional[DataLoader] = None,
          return_loss: bool = False, 
          lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
         ) -> Optional[float]:
    """
    Trains a neural network for one full epoch.

    Args:
        network (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        criterion (torch.nn._Loss): The loss function to minimize.
        train_loader (Optional[torch.utils.data.DataLoader]): DataLoader providing training data. Defaults to None.
        return_loss (bool): If True, returns the average training loss after the epoch.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler to update LR. Defaults to None.

    Returns:
        Optional[float]: The average training loss if return_loss is True, otherwise None.
    """
    
    network.train()
    total_loss = 0

    for (data, target) in train_loader:
        
        
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if not lr_scheduler is None:
            lr_scheduler.step()
            
        
    avg_loss = total_loss / len(train_loader)
    if not lr_scheduler is None:
        print(lr_scheduler.get_last_lr()[0])
    if return_loss:
        return avg_loss


def test(network: Module, 
         criterion: _Loss, 
         valid_loader: DataLoader, 
         means_path: Optional[str] = None, 
         stds_path: Optional[str] = None, 
         DEVICE: torch.device = torch.device('cpu')) -> float:
    """
    Evaluates the model on validation data. Optionally reverses z-score normalization 
    on predictions using provided mean and standard deviation values, restoring them to log2 scale.

    Args:
        network (torch.nn.Module): The trained neural network model.
        criterion (torch.nn._Loss): The loss function to compute validation loss.
        valid_loader (torch.utils.data.DataLoader): DataLoader containing validation inputs and log2-transformed targets.
        means_path (Optional[str]): Path to .npy file with per-feature means (for reversing z-score normalization).
        stds_path (Optional[str]): Path to .npy file with per-feature standard deviations (for reversing z-score normalization).
        DEVICE (torch.device): Device on which computation should run (e.g., CPU or CUDA).

    Returns:
        float: Average validation loss in log2-transformed space.
    """
    
    # Load normalization stats
    if not(means_path is None) and not (stds_path is None):
        means = torch.from_numpy(np.load(means_path)).float().to(DEVICE)
        stds = torch.from_numpy(np.load(stds_path)).float().to(DEVICE)

    network.eval()
    total_val_loss = 0
    
    with torch.inference_mode():
        for (data, target_log2) in valid_loader:


            data = data.to(DEVICE)
            target_log2 = target_log2.to(DEVICE)

            output_z = network(data)
            if not(means_path is None) and not (stds_path is None):
                output_log2 = output_z * stds + means  # reverse z-score to log2
            else:
                output_log2 = output_z

            val_loss = criterion(output_log2, target_log2)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(valid_loader)
    return avg_val_loss

def train_N_epochs(network: Module,
                   optimizer: Optimizer,
                   criterion: _Loss,
                   train_loader: DataLoader,
                   valid_loader: DataLoader,
                   num_epochs: int,
                   verbose: bool = False,
                   checkpoint: bool = True,
                   patience: int = 2,
                   model_path: str = 'best_model',
                   best_valid_loss: float = float('inf'),
                   lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   means_path: Optional[str] = None,
                   stds_path: Optional[str] = None,
                   DEVICE: torch.device = torch.device('cpu')
                   ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Trains a neural network model for a specified number of epochs, while monitoring
    training and validation loss. Supports learning rate scheduling, checkpointing,
    and early stopping.

    Args:
        network (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn._Loss): Loss function for optimization.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        verbose (bool, optional): If True, prints training and validation loss per epoch. Default is False.
        checkpoint (bool, optional): If True, saves best model weights. Default is True.
        patience (int, optional): Epochs to wait for validation loss improvement before stopping. Default is 2.
        model_path (str, optional): Path to save the best model checkpoint. Default is 'best_model'.
        best_valid_loss (float, optional): Initial best validation loss. Default is infinity.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Scheduler to adjust learning rate. Default is None.
        means_path (Optional[str], optional): Path to .npy file with per-feature means (for de-normalization). Default is None.
        stds_path (Optional[str], optional): Path to .npy file with per-feature stds (for de-normalization). Default is None.
        DEVICE (torch.device, optional): Device to run training on (CPU or CUDA). Default is CPU.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Training losses (np.ndarray) over epochs.
            - Validation losses (np.ndarray) over epochs.
            - Learning rates (np.ndarray) over epochs.
        float: The best observed validation loss.
    """
    
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    learning_rates = np.zeros(num_epochs)
    
    current_patience = 0
        
    print('Training model:')
    print(network)
    print(optimizer)
         
    for epoch in range(num_epochs):
        
        avg_train_loss = train(network, optimizer, criterion, 
                               train_loader=train_loader, 
                               return_loss=True, lr_scheduler=lr_scheduler)
        
        avg_val_loss = test(network, criterion, valid_loader, means_path, stds_path, DEVICE = DEVICE)
        
        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss
       
        if lr_scheduler is not None:
            
            learning_rates[epoch] = lr_scheduler.get_last_lr()[0]
        
        if verbose:
            print(f'Epoch {epoch} finished with val_loss: {avg_val_loss} and train_loss: {avg_train_loss}')
            
        # Check if validation loss has improved
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            current_patience = 0
            # Save the model checkpoint if needed
            if checkpoint:
                checkpoint_dict = {'epoch': epoch, 
                                   'network': network.state_dict(), 
                                   'optimizer': optimizer.state_dict(),
                                   'lr_sched': lr_scheduler.state_dict() if lr_scheduler else None, 
                                   'best_valid_loss': best_valid_loss}
                torch.save(checkpoint_dict, f'{model_path}')
        else:
            current_patience += 1
        
        # Check if early stopping criteria met
        if current_patience >= patience:
            print("Early stopping! Validation loss hasn't improved for {} epochs.".format(patience))
            break

        
    return (train_losses[:epoch+1], val_losses[:epoch+1], learning_rates[:epoch+1]), best_valid_loss