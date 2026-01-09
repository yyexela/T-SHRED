"""
Recurrent Neural Network encoders for sequence modeling.
Implements GRU and LSTM encoders compatible with the encoder-decoder architecture.
"""

import torch
import einops
import torch.nn as nn


class GRU(nn.Module):
    """
    GRU encoder for sequence-to-sequence modeling.

    Wraps PyTorch's GRU with dropout and output reshaping for compatibility
    with the encoder-decoder architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the GRU encoder.

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): Hidden state dimension
            num_layers (int): Number of stacked GRU layers
            dropout (float): Dropout probability applied to outputs
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None  # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """
        Forward pass through the GRU encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            dict: Dictionary containing:
                - "sequence_output" (torch.Tensor): All hidden states, shape (batch, 1, seq_len, hidden_size)
                - "final_hidden_state" (torch.Tensor): Final hidden states, shape (batch, 1, num_layers, hidden_size)
                - "output" (torch.Tensor): Last layer's final hidden state, shape (batch, 1, 1, hidden_size)
        """
        out, h_out = self.gru(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")  # encoder_depth

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": h_out[:, :, -1:, :],
        }


class LSTM(nn.Module):
    """
    LSTM encoder for sequence-to-sequence modeling.

    Wraps PyTorch's LSTM with dropout and output reshaping for compatibility
    with the encoder-decoder architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the LSTM encoder.

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): Hidden state dimension
            num_layers (int): Number of stacked LSTM layers
            dropout (float): Dropout probability applied to outputs
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None  # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """
        Forward pass through the LSTM encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            dict: Dictionary containing:
                - "sequence_output" (torch.Tensor): All hidden states, shape (batch, 1, seq_len, hidden_size)
                - "final_hidden_state" (torch.Tensor): Final hidden states, shape (batch, 1, num_layers, hidden_size)
                - "output" (torch.Tensor): Last layer's final hidden state, shape (batch, 1, 1, hidden_size)
        """
        out, (h_out, c_out) = self.lstm(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "h b d -> b 1 h d")  # encoder_depth

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": h_out[:, :, -1:, :],
        }
