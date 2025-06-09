import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, d_model: int, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.gru = None # lazy initialization
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.initialize()

    def initialize(self):
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.d_model), device=self.device)
        out, h_out = self.gru(x, h_0)

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.d_model)
        }

class LSTM(nn.Module):
    def __init__(self, d_model: int, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.d_model), device=self.device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.d_model), device=self.device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.d_model),
            "sindy_loss": None
        }