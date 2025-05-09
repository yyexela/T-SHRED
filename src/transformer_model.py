import torch
import math

import torch.nn as nn
from positional_encoding import PositionalEncoding

# U-Net decoder component
class UNetDecoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.dropout(x)

# TimeSeries_UTransformer with U-Net decoder
class TimeSeries_UTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, sequence_length: int = 5400, dropout: float = 0.1):
        super().__init__()
        self.source_mask = self._generate_square_subsequent_mask(sequence_length)
        self.pos_encoder = PositionalEncoding(d_model, sequence_length, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout, activation=nn.ReLU())
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.input_embedding = nn.Linear(1, d_model)
        self.relu = nn.ReLU()
        self.unet_decoder = UNetDecoder(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input embedding
        x = self.input_embedding(x)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x, self.source_mask)

        # Apply U-Net decoder
        x = self.unet_decoder(x)
        return x

    def _generate_square_subsequent_mask(self, sequence_length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length)) == 0
        return mask

