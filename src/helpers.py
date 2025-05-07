import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import einops
import torch.nn as nn

import torch
import torch.nn as nn

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
    # Ensure input is 4D
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (N, W, H, C), got {tensor.dim()}D tensor")
    
    # Calculate mean and std across all dimensions except channel
    if mean is None:
        mean = tensor.mean(dim=(0, 1, 2), keepdim=True)  # Shape: (1, 1, 1, C)
    if std is None:
        std = tensor.std(dim=(0, 1, 2), keepdim=True)    # Shape: (1, 1, 1, C)
    
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
    # Ensure input is 4D
    if normalized_tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (N, W, H, C), got {normalized_tensor.dim()}")
    
    # Denormalize
    denormalized = normalized_tensor * (std + eps) + mean
    
    return denormalized

def train_model(model, dataloader, sensors, args):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): PyTorch DataLoader instance.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        device (torch.device): Device to train on (e.g. CPU, GPU).
    """
    # Set up model, optimizer, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    model.to(args.device)

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
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

            # TODO: Remove for testing
            if (i+1) % 5 == 0:
                break

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

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