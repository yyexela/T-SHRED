import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import einops
import torch.nn as nn
import random
from src.plots import plot_losses

def generate_sensor_positions(n_sensors: int, max_rows: int, max_cols: int) -> list[tuple[int, int]]:
    random.seed(0)
    return [(random.randint(0, max_rows-1), random.randint(0, max_cols-1)) for _ in range(n_sensors)]

def print_dictionary(hp_dict: dict[str, str], text: str) -> None:
    """
    Print given dictionary

    `hp_dict`: dictionary dictionary to print key and values for
    `text`: text to print before dictionary

    Returns: `None`
    """
    print(text)
    for key in sorted(hp_dict.keys()):
        print(f"> {key}: {hp_dict[key]}")
    print()

    return None

def normalize_pytorch(tensor, dims, mean=None, std=None, eps=1e-8):
    """
    Normalize a tensor across its channel dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, W, H, C)
        mean (torch.Tensor, optional): Pre-computed mean values for each channel
        std (torch.Tensor, optional): Pre-computed standard deviation values for each channel
        eps (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: Normalized tensor of same shape as input
        torch.Tensor: Mean values used for normalization
        torch.Tensor: Standard deviation values used for normalization
    """
    # Calculate mean and std across all dimensions except channel
    if mean is None:
        mean = tensor.mean(dim=dims, keepdim=True)
    if std is None:
        std = tensor.std(dim=dims, keepdim=True)
    
    # Normalize
    normalized = (tensor - mean) / (std + eps)
    
    return normalized, mean, std

def inverse_normalize_pytorch(normalized_tensor, mean, std, eps=1e-8):
    """
    Denormalize a tensor that was previously normalized using normalize_channels.
    
    Args:
        normalized_tensor (torch.Tensor): Normalized tensor of shape (N, W, H, C)
        mean (torch.Tensor): Mean values used for normalization, shape (1, 1, 1, C)
        std (torch.Tensor): Standard deviation values used for normalization, shape (1, 1, 1, C)
        eps (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: Denormalized tensor of same shape as input
    """
    # Denormalize
    denormalized = normalized_tensor * (std + eps) + mean
    
    return denormalized

def evaluate_model(model, test_dl, sensors, args):
    """
    Evaluate a PyTorch model.
    """
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_dl:
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            outputs = model(input_sensors)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows, w=args.data_cols, d=args.d_data)

            # Calculate loss
            test_loss += loss_fn(outputs, labels).item()

        # Average loss
        test_loss /= len(test_dl)

    return test_loss


def train_model(model, train_dl, val_dl, sensors, start_epoch, best_val, best_epoch, train_losses, val_losses, optimizer, args):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dl (DataLoader): PyTorch DataLoader instance for training data.
        val_dl (DataLoader): PyTorch DataLoader instance for validation data.
        sensors (list): List of sensor locations.
        start_epoch (int): Epoch to start training from.
        best_val (float): Best validation loss.
        best_epoch (int): Epoch of best validation loss.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        args (argparse.Namespace): Arguments to use for training.
    """
    # Set up model, optimizer, and loss
    loss_fn = torch.nn.MSELoss()
    model.to(args.device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Calculate training loss
        train_loss = 0.0
        for i, batch in enumerate(train_dl):
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            optimizer.zero_grad()
            outputs = model(input_sensors)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows, w=args.data_cols, d=args.d_data)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # Average loss
        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Calculate validation loss
        val_loss = evaluate_model(model, val_dl, sensors, args)
        val_losses.append(val_loss)

        # Save model to checkpoint if validation loss is lower than best validation loss
        if val_loss < best_val:
            print()
            if args.verbose:
                print(f'Saving model to {args.best_checkpoint_path}, validation loss improved from {best_val:0.4e} to {val_loss:0.4e}, ')
            best_val = val_loss
            best_epoch = epoch+1
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.best_checkpoint_path)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.latest_checkpoint_path)
            print()
        
        # Save model to checkpoint if save_every_n_epochs is reached
        if (epoch + 1) % args.save_every_n_epochs == 0:
            print()
            if args.verbose:
                print(f'Saving model to {args.latest_checkpoint_path}')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.latest_checkpoint_path)
            print()

        # Print loss
        if args.verbose:
            print(f'Epoch {epoch+1}, Training loss: {train_loss:0.4e}, Validation loss: {val_loss:0.4e} (best: {best_val:0.4e})')
        
        # Make plot
        plot_losses(train_losses, val_losses, best_epoch, save=True, fname=f"{args.encoder}_{args.decoder}_{args.dataset}_{args.lr:0.2e}_losses")

    if args.verbose:
        print(f"Training complete, best validation loss: {best_val:0.4e}")
        print()

def calculate_library_dim(latent_dim, poly_order, include_sine):
    dim = 1 # Constant term
    # Polynomial terms (using combinations with replacement)
    current_dim = latent_dim
    dim += current_dim
    if poly_order > 1:
        current_dim = current_dim * (latent_dim + 1) // 2
        dim += current_dim
    if poly_order > 2:
        current_dim = current_dim * (latent_dim + 2) // 3
        dim += current_dim
    if poly_order > 3:
        current_dim = current_dim * (latent_dim + 3) // 4
        dim += current_dim
    if poly_order > 4:
        current_dim = current_dim * (latent_dim + 4) // 5
        dim += current_dim

    if include_sine:
        dim += latent_dim
    return dim

def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    device = z.device # Get device from input tensor
    library = [torch.ones(z.shape[0], device=device)]

    # Add polynomials up to poly_order
    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i] * z[:,j]) # Use element-wise multiplication

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i] * z[:,j] * z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

    # Add sine terms if requested
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)