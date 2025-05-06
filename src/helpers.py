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