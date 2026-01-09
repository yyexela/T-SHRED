"""
SINDy Attention SINDy Loss Transformer module.

Combines SINDy-based attention mechanisms with SINDy loss regularization
for learning sparse, interpretable dynamics in transformer architectures.
"""

import einops
import torch.nn as nn
from sindy_loss_abc import SINDyLoss
from sindy_attention_transformer import SindyAttentionTransformer


class SindyAttentionSindyLossTransformer(SINDyLoss, SindyAttentionTransformer):
    """
    Transformer with both SINDy attention and SINDy loss regularization.

    Combines SINDy-based attention mechanisms (which replace standard attention
    with ODE-based latent space rollouts) with SINDy loss regularization
    (which penalizes deviations from learned sparse dynamics).

    Copied from pytorch:
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        forecast_length: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        strict_symmetry: bool,
        input_length: int,
        hidden_size: int,
        poly_order: int,
        sindy_loss_threshold: float,
        dt: float,
        device: str = "cpu",
    ):
        """
        Initialize the SINDy attention SINDy loss transformer.

        Args:
            d_model (int): Input dimension of the model
            n_heads (int): Number of attention heads
            forecast_length (int): Number of future timesteps to predict
            num_encoder_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout probability
            activation (nn.Module): Activation function for feedforward layers
            layer_norm_eps (float): Epsilon for layer normalization
            norm_first (bool): Whether to apply layer norm before attention
            bias (bool): Whether to use bias in linear layers
            strict_symmetry (bool): Whether to enforce strict symmetry in SINDy coefficients in SINDy attention
            input_length (int): Length of input sequences
            hidden_size (int): Hidden dimension size
            poly_order (int): Polynomial order for SINDy library in SINDy loss
            sindy_loss_threshold (float): Threshold for SINDy coefficient sparsification in SINDy loss
            dt (float): Time step for SINDy derivatives
            device (str): Device to place the model on (default: "cpu")
        """
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            input_length=input_length,
            hidden_size=hidden_size,
            poly_order=poly_order,
            sindy_loss_threshold=sindy_loss_threshold,
            dt=dt,
            forecast_length=forecast_length,
            strict_symmetry=strict_symmetry,
            device=device,
        )

    def forward(
        self,
        src,
        src_mask=None,
        is_causal=True,
    ):
        """
        Forward pass through the SINDy attention SINDy loss transformer.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): Attention mask. Default: None
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full output (batch_size, forecast_length, seq_len, d_model)
                - final_hidden_state: Last timestep (batch_size, forecast_length, d_model)
                - output: Same as sequence_output (batch_size, forecast_length, sequence_length, hidden_size)
                - sindy_loss: SINDy regularization loss value
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

        # Compute SINDy Loss, put forecast dimension into batch dimension
        transformer_output_3d = einops.rearrange(
            transformer_output, "b n s d -> (b n) s d"
        )
        sindy_loss = self.compute_sindy_loss(transformer_output_3d)

        return {
            "sequence_output": transformer_output,
            "final_hidden_state": transformer_output[:, :, -1, :],
            "output": transformer_output,
            "sindy_loss": sindy_loss,
        }
