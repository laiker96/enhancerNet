import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Tuple, List, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(network: torch.nn.Module,
          optimizer: Optimizer,
          criterion: _Loss,
          train_loader: DataLoader,
          return_loss: bool = False,
          lr_scheduler: Optional[Any] = None,
          step_scheduler_per_batch: bool = True) -> Optional[float]:
    """
    Trains the neural network for one epoch.

    Parameters:
        network (torch.nn.Module): Neural network model.
        optimizer (Optimizer): Optimizer.
        criterion (_Loss): Loss function.
        train_loader (DataLoader): DataLoader for training data.
        return_loss (bool): If True, returns average training loss.
        lr_scheduler (optional): Learning rate scheduler.
        step_scheduler_per_batch (bool): Whether to call scheduler.step() after every batch.

    Returns:
        float or None: Average training loss if return_loss is True.
    """
    network.train()
    total_loss = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if lr_scheduler is not None and step_scheduler_per_batch:
            lr_scheduler.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss if return_loss else None


def test(network: torch.nn.Module,
         criterion: _Loss,
         valid_loader: DataLoader,
         means_path: Optional[str] = None,
         stds_path: Optional[str] = None,
         limit_obs: bool = False,
         n_val_examples: int = 10000) -> float:
    """
    Evaluates the model on the validation set.

    If means_path and stds_path are provided, output is Z-score denormalized back to log2 scale.

    Parameters:
        network (torch.nn.Module): Model to evaluate.
        criterion (_Loss): Loss function.
        valid_loader (DataLoader): DataLoader for validation data.
        means_path (str, optional): Path to .npy file with means for each output feature.
        stds_path (str, optional): Path to .npy file with stds for each output feature.
        limit_obs (bool): Whether to limit the number of validation examples.
        n_val_examples (int): Number of validation examples if limit_obs is True.

    Returns:
        float: Average validation loss.
    """
    if means_path and stds_path:
        means = torch.tensor(np.load(means_path), dtype=torch.float32).to(DEVICE)
        stds = torch.tensor(np.load(stds_path), dtype=torch.float32).to(DEVICE)
    else:
        means = stds = None

    network.eval()
    total_val_loss = 0
    batch_size = valid_loader.batch_size

    with torch.inference_mode():
        for batch_i, (data, target_log2) in enumerate(valid_loader):
            if limit_obs and (batch_i * batch_size > n_val_examples):
                break

            data = data.to(DEVICE)
            target_log2 = target_log2.to(DEVICE)

            output_z = network(data)

            if means is not None and stds is not None:
                output_log2 = output_z * stds + means
            else:
                output_log2 = output_z  # No denormalization

            val_loss = criterion(output_log2, target_log2)
            total_val_loss += val_loss.item()

    return total_val_loss / len(valid_loader)


def train_N_epochs(network: torch.nn.Module,
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
                   lr_scheduler: Optional[Any] = None,
                   means_path: Optional[str] = None,
                   stds_path: Optional[str] = None) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Trains the model over multiple epochs with optional early stopping, checkpointing, and learning rate scheduling.

    Supports both per-batch and per-epoch schedulers.

    Parameters:
        network (torch.nn.Module): Model to train.
        optimizer (Optimizer): Optimizer.
        criterion (_Loss): Loss function.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        verbose (bool): Whether to print loss each epoch.
        checkpoint (bool): If True, saves model when validation improves.
        patience (int): Early stopping patience.
        model_path (str): Where to save best model.
        best_valid_loss (float): Starting best val loss.
        lr_scheduler (optional): Scheduler.
        means_path (str, optional): Mean file for Z-score reversal.
        stds_path (str, optional): Std file for Z-score reversal.

    Returns:
        Tuple: (train_losses, val_losses, learning_rates), best_valid_loss
    """

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    learning_rates = np.zeros(num_epochs)
    current_patience = 0
    print('Training model:')
    print(network)
    print(optimizer)

    # Identify scheduler stepping behavior
    batch_schedulers = (
        torch.optim.lr_scheduler.OneCycleLR,
        torch.optim.lr_scheduler.CyclicLR,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    )
    is_batch_scheduler = isinstance(lr_scheduler, batch_schedulers)

    for epoch in range(num_epochs):
        avg_train_loss = train(network, optimizer, criterion,
                               train_loader=train_loader,
                               return_loss=True,
                               lr_scheduler=lr_scheduler,
                               step_scheduler_per_batch=is_batch_scheduler)

        avg_val_loss = test(network, criterion, valid_loader,
                            means_path=means_path, stds_path=stds_path)

        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss

        if not is_batch_scheduler and lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(avg_val_loss)
            else:
                lr_scheduler.step()

        if lr_scheduler is not None:
            learning_rates[epoch] = lr_scheduler.optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')

        # Check for improvement
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            current_patience = 0

            if checkpoint:
                checkpoint_dict = {
                    'epoch': epoch,
                    'network': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict() if lr_scheduler else None,
                    'best_valid_loss': best_valid_loss
                }
                torch.save(checkpoint_dict, model_path)
        else:
            current_patience += 1

        if current_patience >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return (train_losses[:epoch+1], val_losses[:epoch+1], learning_rates[:epoch+1]), best_valid_loss


