import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import einops
import torch.nn as nn

import torch
import torch.nn as nn

def train_model(model, dataloader, args):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): PyTorch DataLoader instance.
        num_epochs (int): Number of epochs to train for.
        lr (float): Learning rate.
        device (torch.device): Device to train on (e.g. CPU, GPU).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    model.to(args.device)

    for epoch in range(args.epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Reshape inputs from (batch, window_length, n_sensors, d_data) to (batch, window_length, n_sensors*d_data) using eigops
            inputs = einops.rearrange(inputs, 'b w n d -> b w (n d)')

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = einops.rearrange(outputs, 'b (r w) -> b r w', b=inputs.shape[0], r=args.data_rows, w=args.data_cols)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

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