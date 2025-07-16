import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Tuple, List

def train(network: torch.nn.Module, optimizer: Optimizer, 
          criterion: _Loss, 
          limit_obs: bool = False, 
          train_loader: Optional[DataLoader] = None ,
          n_train_examples: int = 10000, 
          return_loss: bool = False, 
          lr_scheduler = None) -> Optional[float]:
    """
    Trains the neural network model.

    Parameters:
        - network (torch.nn.Module): The neural network model.
        - optimizer (Optimizer): The optimizer for the neural network.
        - criterion (_Loss): The loss function.
        - limit_obs (bool): Whether to limit the training examples for faster training (default: False).
        - train_loader (Optional[DataLoader]): DataLoader class with the training examples (default: None).
        - n_train_examples (int): The number of training examples to consider if limit_obs is True (default: 10000).
        - return_loss (bool): Whether to return the average training loss(default: False).

    Returns:
        - float or None: The average training loss if return_loss is True, else None.
    """
    
    network.train()
    batch_size = train_loader.batch_size
    total_loss = 0

    for batch_i, (data, target) in enumerate(train_loader):
        
        # Limit train data for faster computation
        if limit_obs and (batch_i * batch_size >= n_train_examples):
            break
        
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if not lr_scheduler is None:
            lr_scheduler.step()
            #print(lr_scheduler.get_last_lr()[0])
            
        
    avg_loss = total_loss / len(train_loader)
    if not lr_scheduler is None:
        print(lr_scheduler.get_last_lr()[0])
    if return_loss:
        return avg_loss

def test(network: torch.nn.Module, criterion: _Loss, 
         valid_loader: DataLoader, limit_obs: bool = False, 
         n_val_examples: int = 10000) -> float:
    """
    Tests the model.

    Parameters:
        - network (torch.nn.Module): The neural network model.
        - criterion (_Loss): The loss function.
        - valid_loader (DataLoader): DataLoader class with the test examples.

    Returns:
        - float: The validation average loss across batches.
    """
    
    network.eval()
    total_val_loss = 0
    batch_size = valid_loader.batch_size
    
    with torch.inference_mode():
        for batch_i, (data, target) in enumerate(valid_loader):
            # Limit train data for faster computation
            if limit_obs and (batch_i * batch_size > n_val_examples):
                break
            output = network(data)
            val_loss = criterion(output, target)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(valid_loader)
    return avg_val_loss

def train_N_epochs(network: torch.nn.Module, optimizer: Optimizer, 
                   criterion: _Loss, train_loader: DataLoader, 
                   valid_loader: DataLoader, num_epochs: int, 
                   verbose: bool = False, checkpoint: bool = True, 
                   patience: int = 2, model_path: str = 'best_model', 
                   best_valid_loss: float = float('inf'), 
                   lr_scheduler = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a neural network model for a specified number of epochs, monitoring and recording
    average training and validation losses for each epoch.

    Parameters:
        - network (torch.nn.Module): The neural network model to be trained.
        - optimizer (Optimizer): The optimizer used for updating model weights.
        - criterion (_Loss): The loss function used to compute the loss between predictions and targets.
        - train_loader (DataLoader): DataLoader containing training data.
        - valid_loader (DataLoader): DataLoader containing validation (or test) data.
        - num_epochs (int): The number of training epochs (iterations over the training dataset).
        - verbose (bool, optional): If True, print progress information during training (default is False).
        - checkpoint (bool, optional): If True, save the model's state when validation loss improves (default is True).
        - patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping (default is 2).
        - model_path (str, optional): The path prefix where the best model checkpoint will be saved (default is 'best_model').

    Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
          - The first array contains the average training losses for each epoch.
          - The second array contains the average validation losses for each epoch.

    Notes:
        - Early stopping is applied if the validation loss does not improve for the specified patience epochs.
    """

    
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    learning_rates = np.zeros(num_epochs)
    # Initialize early stopping parameters
    #best_valid_loss = float('inf')
    current_patience = 0
    #if lr_scheduler is None:

        #lr_scheduler = OneCycleLR(optimizer, max_lr = 1e-3, epochs = num_epochs, steps_per_epoch = len(train_loader))
        #lr_scheduler = OneCycleLR(optimizer, max_lr = 1e-4, epochs = num_epochs, steps_per_epoch = len(train_loader))

        
    print('Training model:')
    print(network)
    print(optimizer)
         
    for epoch in range(num_epochs):
        
        avg_train_loss = train(network, optimizer, criterion, 
                               train_loader=train_loader, 
                               return_loss=True, lr_scheduler=lr_scheduler)
        
        avg_val_loss = test(network, criterion, valid_loader)
        
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


