import torch
import einops
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 num_layers:int,
                 dropout:float,
                 device:str = 'cpu',
                 **kwargs):
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
        out, h_out = self.gru(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, 'b s d -> b 1 s d')
        h_out = einops.rearrange(h_out, 'h b d -> b 1 h d') # encoder_depth

        return {
            "sequence_output": out, # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out, # [batch_size, 1, encoder_depth, d_model]
            "output": h_out[:,:,-1:,:],
        }

class LSTM(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 num_layers:int,
                 dropout:float,
                 device:str = 'cpu',
                 **kwargs):
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
        out, (h_out, c_out) = self.lstm(x)

        out = self.dropout(out)
        h_out = self.dropout(h_out)
        out = einops.rearrange(out, 'b s d -> b 1 s d')
        h_out = einops.rearrange(h_out, 'h b d -> b 1 h d') # encoder_depth

        return {
            "sequence_output": out, # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out, # [batch_size, 1, encoder_depth, d_model]
            "output": h_out[:,:,-1:,:],
        }