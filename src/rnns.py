import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size:int = 3, hidden_size:int = 64, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)

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
        device = next(self.parameters()).device
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        out, h_out = self.gru(x, h_0)

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size)
        }

class LSTM(nn.Module):
    def __init__(self, input_size:int = 3, hidden_size:int = 64, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
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
        device = next(self.parameters()).device
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size),
            "sindy_loss": None
        }