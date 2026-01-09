"""
SINDy Loss MOE RNN modules.

Implements MOE-GRU, MOE-LSTM, and MOE-MLP encoders with SINDy loss regularization for
learning interpretable sparse dynamics in mixed expert recurrent neural networks.
"""

import torch
import einops
import torch.nn as nn
from moe_rnns import MOEGRU, MOELSTM, MOEMLP
from sindy_loss_abc import SINDyLoss


class SINDyLossMOEGRU(SINDyLoss, MOEGRU):
    """
    MOE-GRU encoder with SINDy loss regularization.

    Combines a standard MOE-GRU encoder with SINDy-based regularization that
    encourages the hidden state dynamics to follow a sparse polynomial ODE.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
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
            n_experts (int): Number of SINDy expert layers
            forecast_length (int): Number of timesteps to forecast
            strict_symmetry (bool): If True, enforce symmetric SINDy coefficients
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
            n_experts=n_experts,
            forecast_length=forecast_length,
            strict_symmetry=strict_symmetry,
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
        # Normal GRU forward
        out, h_out = self.gru(x)

        # SINDy loss
        sindy_loss = self.compute_sindy_loss(out)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": combined,
            "sindy_loss": sindy_loss,
        }

class SINDyLossMOELSTM(SINDyLoss, MOELSTM):
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
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
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
            n_experts (int): Number of SINDy expert layers
            forecast_length (int): Number of timesteps to forecast
            strict_symmetry (bool): If True, enforce symmetric SINDy coefficients
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
            n_experts=n_experts,
            forecast_length=forecast_length,
            strict_symmetry=strict_symmetry,
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
        # Normal LSTM forward
        out, (h_out, c_out) = self.lstm(x)

        # SINDy loss
        sindy_loss = self.compute_sindy_loss(out)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3) # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": combined,
            "sindy_loss": sindy_loss,
        }

class SINDyLossMOEMLP(SINDyLoss, MOEMLP):
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
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
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
            n_experts (int): Number of SINDy expert layers
            forecast_length (int): Number of timesteps to forecast
            strict_symmetry (bool): If True, enforce symmetric SINDy coefficients
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
            n_experts=n_experts,
            forecast_length=forecast_length,
            strict_symmetry=strict_symmetry,
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
        out = self.mlp(x)

        # SINDy loss
        sindy_loss = self.compute_sindy_loss(out)

        # SINDy forward all experts
        sindy_outputs = [expert(out[:, -1, :]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return {
            "sequence_output": out,
            "output": combined,
            "sindy_loss": sindy_loss,
        }
