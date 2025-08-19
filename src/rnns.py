import torch
import einops
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, dropout:float, device:str = 'cpu', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.initialize()

    def initialize(self):
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, h_out = self.gru(x, h_0)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        h_out = einops.rearrange(h_out, 'r b d -> b r d')

        return {
            "sequence_output": out, # [batch_size, sequence_length, d_model]
            "final_hidden_state": h_out # [batch_size, rollout, d_model]
        }

class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, dropout:float, device:str = 'cpu', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        h_out = einops.rearrange(h_out, 'r b d -> b r d')

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "sindy_loss": None
        }