import math
import copy
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    source: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """
    def __init__(self, d_model: int, sequence_length: int = 5400, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(sequence_length, 1, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x