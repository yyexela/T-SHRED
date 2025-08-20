import einops
import torch.nn as nn
from sindy_loss_abc import SINDyLoss
from sindy_attention_transformer import SindyAttentionTransformer

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html

class SindyAttentionSindyLossTransformer(SINDyLoss, SindyAttentionTransformer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation : nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        poly_order: int,
        sindy_loss_threshold: float,
        dt: float,
        device='cpu',
    ):
        super().__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, bias=bias, input_length=input_length, hidden_size=hidden_size, poly_order=poly_order, dt=dt, sindy_loss_threshold=sindy_loss_threshold, device=device)

    def forward(
        self,
        src,
        src_mask=None,
        is_causal=True,
    ):
        x_embedded, _ = self.input_embedding(src) # Shape: (batch_size, seq_len, d_model)

        x_pos_encoded = self.pos_encoder(x_embedded) # Shape: (batch_size, seq_len, d_model)

        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=is_causal,
        )

        # Compute SINDy Loss
        sindy_loss = self.compute_sindy_loss(transformer_output)

        # reshape output
        transformer_output = einops.rearrange(transformer_output, 'b s d -> b 1 s d')

        return {
            "sequence_output": transformer_output, # [batch_size, rollout, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, :, -1, :], # Last timestep [batch_size, rollout, d_model]
            "output": transformer_output, # [batch_size, rollout, sequence_length, d_model]
            "sindy_loss": sindy_loss
        }