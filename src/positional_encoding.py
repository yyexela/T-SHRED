"""
Positional encoding for transformer models.
Implements sinusoidal position embeddings to provide sequence order information.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds position-dependent sinusoidal signals to input embeddings to provide
    sequence position information. Uses sine for even dimensions and cosine
    for odd dimensions with exponentially decreasing frequencies.

    Reference: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """

    def __init__(
        self, d_model: int, sequence_length: int, dropout: float, device: str = "cpu"
    ):
        """
        Initialize the positional encoding layer.

        Args:
            d_model (int): Embedding dimension (must match input feature size)
            sequence_length (int): Maximum sequence length to encode
            dropout (float): Dropout probability applied after adding positional encoding
            device (str): Device to place the encoding buffer on (default: "cpu")
        """
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(1, sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pos_encoding[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pos_encoding.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Input with positional encoding added and dropout applied,
                same shape as input (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        x = self.dropout(x)
        return x
