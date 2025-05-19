import math
import copy
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from positional_encoding import PositionalEncoding

class TRANSFORMER(nn.Module):
    """
    Standard Transformer Encoder model using nn.TransformerEncoderLayer.
    Uses GRU for input embedding as per the original user code structure.
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, # nhead=16 was high for d_model=128, using 8
                 num_encoder_layers: int = 1, dim_feedforward: int = 256, # Often 4*d_model
                 dropout: float = 0.2, activation = nn.GELU(),
                 window_length: int = 10, hidden_size: int = 10, device:str='cpu'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.window_length = window_length
        self.hidden_size = hidden_size
        self.device = device

        # --- Standard Transformer Components ---
        # Define a single encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True, # Important: assumes input shape (batch, seq, feature)
            norm_first=False   # Standard: Apply norm after Add; True applies before
        )
        # Stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_size) if num_encoder_layers > 1 else None # Optional final norm
        )
        # --- End Standard Transformer Components ---

        self.pos_encoder = None
        self.input_embedding = None

        self.output_size = d_model # Output dim is the model's hidden dim

        self.initialize()

    def initialize(self):
        """Initialize components that depend on input dimensions and sequence length."""
        # Ensure d_model is divisible by nhead
        if self.hidden_size % self.nhead != 0:
             raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by nhead ({self.nhead})")

        # Initialize Positional Encoding with correct sequence length awareness
        # Note: max_sequence_length in PositionalEncoding should be >= lags
        self.pos_encoder = PositionalEncoding(
            d_model=self.hidden_size,
            sequence_length=self.window_length + 10, # Provide some buffer
            dropout=self.dropout
        )

        # Initialize Input Embedding (using GRU as per original structure)
        self.input_embedding = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_size, # GRU output matches d_model
            num_layers=2,                 # Example: 2 GRU layers for embedding
            batch_first=True,
            dropout=self.dropout if self.num_encoder_layers > 1 else 0.0 # Dropout between GRU layers
        )

    def _generate_square_subsequent_mask(self, sequence_length: int, device) -> torch.Tensor:
        """Generates an upper-triangular matrix mask for causal attention."""
        # Creates a mask where future positions are masked (-inf or True)
        # torch.triu: upper triangle of a matrix. diagonal=1 means diagonal is 0 (not masked)
        mask = torch.triu(torch.full((sequence_length, sequence_length), float('-inf'), device=device), diagonal=1)
        # For nn.TransformerEncoderLayer's 'src_mask', float('-inf') is standard.
        # If 'attn_mask' in nn.MultiheadAttention is used directly, it often expects boolean (True means ignore)
        return mask # Shape: [sequence_length, sequence_length]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Dictionary containing sequence output and final hidden state
        """
        if self.input_embedding is None or self.pos_encoder is None:
            raise RuntimeError("Model must be initialized before calling forward.")

        # 1. Input Embedding
        # GRU returns output sequence and hidden state(s)
        # We only need the output sequence here
        x_embedded, _ = self.input_embedding(x) # Shape: (batch_size, seq_len, d_model)

        # 2. Add Positional Encoding
        x_pos_encoded = self.pos_encoder(x_embedded) # Shape: (batch_size, seq_len, d_model)

        # 3. Generate Causal Mask
        # Mask prevents attention to future positions. Shape: [seq_len, seq_len]
        # Needs to be on the same device as the input tensor 'x'.
        causal_mask = self._generate_square_subsequent_mask(x_pos_encoded.size(1), x_pos_encoded.device)

        # 4. Apply Transformer Encoder Layers
        # The mask ensures causality.
        transformer_output = self.transformer_encoder(
            src=x_pos_encoded,
            mask=causal_mask,
            src_key_padding_mask=None # Optional: for masking padded elements
        ) # Shape: (batch_size, seq_len, d_model)

        # 5. Apply dropout
        transformer_output = self.dropout_layer(transformer_output)

        # 5. Prepare Output
        return {
            "sequence_output": transformer_output, # [batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, -1, :], # Last timestep [batch_size, d_model]
            "sindy_loss": None
        }