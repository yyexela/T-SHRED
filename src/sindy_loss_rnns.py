"""
SINDy Loss RNN modules.

Implements GRU, LSTM, and MLP encoders with SINDy loss regularization for
learning interpretable sparse dynamics in mixed expert recurrent neural networks.
"""

import torch
import einops
import torch.nn as nn
from rnns import GRU, LSTM, MLPEncoder
from sindy_loss_abc import SINDyLoss


class SINDyLossGRU(SINDyLoss, GRU):
    """
    GRU encoder with SINDy loss regularization.

    Combines a standard GRU encoder with SINDy-based regularization that
    encourages the hidden state dynamics to follow a sparse polynomial ODE.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        poly_order: int,
        dt: float,
        sindy_loss_threshold: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the SINDy Loss GRU.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of GRU hidden state
            num_layers (int): Number of stacked GRU layers
            dropout (float): Dropout probability between GRU layers
            poly_order (int): Polynomial order for SINDy library
            dt (float): Time step for SINDy derivatives
            sindy_loss_threshold (float): Threshold for coefficient sparsification
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            poly_order=poly_order,
            dt=dt,
            sindy_loss_threshold=sindy_loss_threshold,
        )

    def forward(self, x):
        """
        Forward pass through the SINDy Loss GRU.

        Processes input through GRU layers and computes SINDy loss
        based on how well hidden state transitions follow learned dynamics.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full sequence (batch_size, 1, seq_len, hidden_size)
                - final_hidden_state: Last hidden state (batch_size, 1, 1, hidden_size)
                - output: Same as final_hidden_state (batch_size, 1, 1, hidden_size)
                - sindy_loss: SINDy regularization loss value
        """
        out, h_out = self.gru(x)
        h_out = h_out[-1:]

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "s b d -> b 1 s d")

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": h_out,
            "sindy_loss": sindy_loss,
        }


class SINDyLossLSTM(SINDyLoss, LSTM):
    """
    LSTM encoder with SINDy loss regularization.

    Combines a standard LSTM encoder with SINDy-based regularization that
    encourages the hidden state dynamics to follow a sparse polynomial ODE.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        poly_order: int,
        dt: float,
        sindy_loss_threshold: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the SINDy Loss LSTM.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of LSTM hidden state
            num_layers (int): Number of stacked LSTM layers
            dropout (float): Dropout probability between LSTM layers
            poly_order (int): Polynomial order for SINDy library
            dt (float): Time step for SINDy derivatives
            sindy_loss_threshold (float): Threshold for coefficient sparsification
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            poly_order=poly_order,
            dt=dt,
            sindy_loss_threshold=sindy_loss_threshold,
        )

    def forward(self, x):
        """
        Forward pass through the SINDy Loss LSTM.

        Processes input through LSTM layers and computes SINDy loss
        based on how well hidden state transitions follow learned dynamics.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full sequence (batch_size, 1, seq_len, hidden_size)
                - final_hidden_state: Last hidden state (batch_size, 1, 1, hidden_size)
                - output: Same as final_hidden_state (batch_size, 1, 1, hidden_size)
                - sindy_loss: SINDy regularization loss value
        """
        # Initialize hidden and cell
        out, (h_out, c_out) = self.lstm(x)
        h_out = h_out[-1:]

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "s b d -> b 1 s d")

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": h_out,
            "sindy_loss": sindy_loss,
        }


class SINDyLossMLP(SINDyLoss, MLPEncoder):
    """
    MLP encoder with SINDy loss regularization.

    Combines a standard MLP encoder with SINDy-based regularization that
    encourages the hidden state dynamics to follow a sparse polynomial ODE.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        poly_order: int,
        dt: float,
        sindy_loss_threshold: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the SINDy Loss MLP.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of MLP hidden state
            num_layers (int): Number of stacked MLP layers
            dropout (float): Dropout probability between MLP layers
            poly_order (int): Polynomial order for SINDy library
            dt (float): Time step for SINDy derivatives
            sindy_loss_threshold (float): Threshold for coefficient sparsification
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            poly_order=poly_order,
            dt=dt,
            sindy_loss_threshold=sindy_loss_threshold,
        )

    def forward(self, x):
        """
        Forward pass through the SINDy Loss LSTM.

        Processes input through LSTM layers and computes SINDy loss
        based on how well hidden state transitions follow learned dynamics.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full sequence (batch_size, 1, seq_len, hidden_size)
                - final_hidden_state: Last hidden state (batch_size, 1, 1, hidden_size)
                - output: Same as final_hidden_state (batch_size, 1, 1, hidden_size)
                - sindy_loss: SINDy regularization loss value
        """
        out = self.model(x)

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        return {
            "sequence_output": out,
            "final_hidden_state": out,
            "output": out[:, :, -1:, :],
            "sindy_loss": sindy_loss,
        }
