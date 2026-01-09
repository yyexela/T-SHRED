"""
Decoder architectures for reconstructing full fields from latent representations.
Implements MLP and CNN decoders for the encoder-decoder framework.
"""

import einops
import numpy as np
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) decoder.

    Creates a feedforward neural network with logarithmically spaced layer sizes
    between the input and output dimensions. Uses ReLU activations between layers
    and applies dropout after the final layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
        device: str = "cpu",
    ):
        """
        Initialize the MLP decoder.

        Args:
            in_dim (int): Input dimension of the MLP
            out_dim (int): Output dimension of the MLP
            n_layers (int): Number of linear layers in the network
            dropout (float): Dropout probability applied after the final layer
            device (str): Device to place the model on (default: "cpu")
        """
        super(MLP, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(
            np.logspace(
                np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers + 1, dtype=int
            ).tolist()
        )
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx != (len(sizes) - 2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        """
        Forward pass through the MLP decoder.

        Args:
            x (dict): Dictionary containing:
                - "output" (torch.Tensor): Input tensor of shape (batch, forecast_length, sequence_length, hidden_dim)
                - "sindy_loss" (torch.Tensor, optional): SINDy loss tensor to pass through

        Returns:
            dict: Dictionary containing:
                - "output" (torch.Tensor): Output tensor of shape (batch, forecast_length, sequence_length, out_dim)
                - "sindy_loss" (torch.Tensor or None): Passed through SINDy loss tensor
        """
        sindy_loss = x.get("sindy_loss", None)
        x = x["output"]
        out = self.model(x)
        out = self.dropout(out)
        return {"output": out, "sindy_loss": sindy_loss}


class CNN(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) decoder.

    Creates a convolutional network with logarithmically spaced channel sizes
    between the input and output dimensions. Uses 1D convolutions with kernel
    size 3 and padding 1, ReLU activations between layers, and applies dropout
    after the final layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
        device: str = "cpu",
    ):
        """
        Initialize the CNN decoder.

        Args:
            in_dim (int): Input dimension of the CNN
            out_dim (int): Output dimension of the CNN
            n_layers (int): Number of convolutional layers in the network
            dropout (float): Dropout probability applied after the final layer
            device (str): Device to place the model on (default: "cpu")
        """
        super().__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(
            np.logspace(
                np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers + 1, dtype=int
            ).tolist()
        )
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

        # Define model layers
        self.layers = []
        for idx in range(len(sizes) - 1):
            self.layers.append(
                nn.Conv1d(sizes[idx], sizes[idx + 1], kernel_size=3, padding=1)
            )
            if idx != (len(sizes) - 2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        """
        Forward pass through the CNN decoder.

        Args:
            x (dict): Dictionary containing:
                - "output" (torch.Tensor): Input tensor of shape (batch, forecast_length, sequence_length, hidden_dim)
                - "sindy_loss" (torch.Tensor, optional): SINDy loss tensor to pass through

        Returns:
            dict: Dictionary containing:
                - "output" (torch.Tensor): Output tensor of shape (batch, forecast_length, sequence_length, out_dim)
                - "sindy_loss" (torch.Tensor or None): Passed through SINDy loss tensor
        """
        sindy_loss = x.get("sindy_loss", None)
        x = x["output"]

        batch_size, forecast_length, sequence_length, hidden_dim = x.shape
        x = einops.rearrange(
            x, "b f s d -> b d (f s)", f=forecast_length, s=sequence_length
        )
        out = self.model(x)
        out = self.dropout(out)  # want: batch forecast seq_len (rows cols dim)
        out = einops.rearrange(
            out, "b o (f s) -> b f s o", f=forecast_length, s=sequence_length
        )
        return {"output": out, "sindy_loss": sindy_loss}
