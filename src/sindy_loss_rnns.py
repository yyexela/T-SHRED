import torch
import einops
import torch.nn as nn
from rnns import GRU, LSTM
from sindy_loss_abc import SINDyLoss

class SINDyLossGRU(SINDyLoss, GRU):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 num_layers:int,
                 dropout:float,
                 poly_order: int,
                 dt: float, # Time step for SINDy derivatives
                 sindy_loss_threshold: float,
                 device:str = 'cpu',
                 **kwargs
                ):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device, poly_order=poly_order, dt=dt, sindy_loss_threshold=sindy_loss_threshold)

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, h_out = self.gru(x, h_0)

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        h_out = einops.rearrange(h_out, 'r b d -> b r d')

        return {
            "sequence_output": out, # [batch_size, sequence_length, d_model]
            "final_hidden_state": h_out, # [batch_size, rollout, d_model]
            "sindy_loss": sindy_loss
        }

class SINDyLossLSTM(SINDyLoss, LSTM):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 num_layers:int,
                 dropout:float,
                 poly_order: int,
                 dt: float, # Time step for SINDy derivatives
                 sindy_loss_threshold: float,
                 device:str = 'cpu',
                 **kwargs
                ):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device, poly_order=poly_order, dt=dt, sindy_loss_threshold=sindy_loss_threshold)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        h_out = einops.rearrange(h_out, 'r b d -> b r d')

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "sindy_loss": sindy_loss
        }
