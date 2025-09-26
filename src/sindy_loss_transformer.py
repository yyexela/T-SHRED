import torch
import einops
import torch.nn as nn
from sindy_loss_abc import SINDyLoss
from vanilla_transformer import Transformer
from positional_encoding import PositionalEncoding
from typing import Optional

class SINDyLossTransformer(SINDyLoss, Transformer):
    """
    Transformer model with additional SINDy loss for learning sparse dynamics.
    This model implements a standard transformer encoder with an additional SINDy component 
    that is used to regularize the latent dynamics.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        hidden_size: int,
        input_length: int,
        num_encoder_layers: int,
        poly_order: int,
        dt: float,
        sindy_loss_threshold: float,
        activation: nn.Module,
        bias: bool,
        layer_norm_eps: float,
        norm_first: bool,
        device: str = 'cpu',
    ):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, hidden_size=hidden_size, input_length=input_length, num_encoder_layers=num_encoder_layers, poly_order=poly_order, dt=dt, sindy_loss_threshold=sindy_loss_threshold, activation=activation, bias=bias, layer_norm_eps=layer_norm_eps, norm_first=norm_first, device=device)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> dict:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, d_model)
        Returns:
            Dictionary containing:
                - sequence_output: Output tensor of shape (batch_size, sequence_length, hidden_size)
                - final_hidden_state: Last timestep hidden state (batch_size, hidden_size)
                - sindy_loss: SINDy regularization loss if training (or None if not)
        """
        # Embed input
        x_embedded = self.input_embedding(src)
        
        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded) # Shape: (batch_size, seq_len, d_model)

        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=is_causal,
        )

        sindy_loss = self.compute_sindy_loss(transformer_output[:,-1:,:])

        transformer_output = einops.rearrange(transformer_output, 'b s d -> b 1 s d')

        return {
            "sequence_output": transformer_output, # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, :, -1, :], # Last timestep [batch_size, forecast_length, d_model]
            "output": transformer_output, # [batch_size, forecast_length, sequence_length, d_model]
            "sindy_loss": sindy_loss
        }