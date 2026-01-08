import torch
import einops
import torch.nn as nn
from rnns import GRU, LSTM
from sindy_loss_abc import SINDyLoss


class SINDyLossGRU(SINDyLoss, GRU):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        poly_order: int,
        dt: float,  # Time step for SINDy derivatives
        sindy_loss_threshold: float,
        device: str = "cpu",
        **kwargs,
    ):
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
        Forward pass through the GRU model.
        """
        out, h_out = self.gru(x)
        h_out = h_out[-1:]

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, "b s d -> b 1 s d")
        h_out = einops.rearrange(h_out, "s b d -> b 1 s d")

        return {
            "sequence_output": out,  # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out,  # [batch_size, sequence_length, d_model]
            "output": h_out,
            "sindy_loss": sindy_loss,
        }


class SINDyLossLSTM(SINDyLoss, LSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        poly_order: int,
        dt: float,  # Time step for SINDy derivatives
        sindy_loss_threshold: float,
        device: str = "cpu",
        **kwargs,
    ):
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
        Forward pass through the LSTM model.
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
