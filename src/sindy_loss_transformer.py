"""
SINDy Loss Transformer module.

Implements a transformer encoder with SINDy loss regularization for
learning interpretable sparse dynamics in attention-based models.
"""

import torch
import einops
import torch.nn as nn
from sindy_loss_abc import SINDyLoss
from vanilla_transformer import Transformer
from positional_encoding import PositionalEncoding
from typing import Optional


class SINDyLossTransformer(SINDyLoss, Transformer):
    """
    Transformer encoder with SINDy loss regularization.

    Combines a standard transformer encoder with SINDy-based regularization
    that encourages the learned representations to follow sparse polynomial ODEs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
        device: str = "cpu",
    ):
        """
        Initialize the SINDy Loss Transformer.

        Args:
            d_model (int): Input dimension of the model
            n_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout probability
            hidden_size (int): Hidden dimension size
            input_length (int): Length of input sequences
            num_encoder_layers (int): Number of transformer encoder layers
            poly_order (int): Polynomial order for SINDy library
            dt (float): Time step for SINDy derivatives
            sindy_loss_threshold (float): Threshold for coefficient sparsification
            activation (nn.Module): Activation function for feedforward layers
            bias (bool): Whether to use bias in linear layers
            layer_norm_eps (float): Epsilon for layer normalization
            norm_first (bool): Whether to apply layer norm before attention
            device (str): Device to place the model on (default: "cpu")
        """
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            hidden_size=hidden_size,
            input_length=input_length,
            num_encoder_layers=num_encoder_layers,
            poly_order=poly_order,
            dt=dt,
            sindy_loss_threshold=sindy_loss_threshold,
            activation=activation,
            bias=bias,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            device=device,
        )

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
                - sequence_output: Output tensor of shape (batch_size, 1, sequence_length, hidden_size)
                - final_hidden_state: Last timestep hidden state (batch_size, 1, hidden_size)
                - output: Same as final_hidden_state (batch_size, 1, sequence_length, hidden_size)
                - sindy_loss: SINDy regularization loss if training (or None if not)
        """
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(x_embedded)

        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=is_causal,
        )

        sindy_loss = self.compute_sindy_loss(transformer_output)

        transformer_output = einops.rearrange(transformer_output, "b s d -> b 1 s d")

        return {
            "sequence_output": transformer_output,
            "final_hidden_state": transformer_output[:, :, -1, :],
            "output": transformer_output,
            "sindy_loss": sindy_loss,
        }
