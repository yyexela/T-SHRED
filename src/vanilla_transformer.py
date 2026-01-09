"""
Vanilla Transformer module.

Implements standard transformer encoder components including multi-head
attention, encoder layers, and the full transformer architecture.
"""

import copy
import torch
import einops
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from positional_encoding import PositionalEncoding
from helpers import _get_clones


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention mechanism.

    Implements scaled dot-product attention with multiple heads,
    supporting both same and different query/key/value dimensions.

    Copied from pytorch:
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        dtype: torch.dtype,
        device: str = "cpu",
    ):
        """
        Initialize the MultiHeadAttention module.

        Args:
            E_q (int): Size of embedding dimension for query
            E_k (int): Size of embedding dimension for key
            E_v (int): Size of embedding dimension for value
            E_total (int): Total embedding dimension of combined heads post input projection.
                Each head has dimension E_total // n_heads
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability for attention weights
            bias (bool): Whether to add bias to input/output projections
            dtype (torch.dtype): Data type for parameters
            device (str): Device to place the model on (default: "cpu")

        Raises:
            AssertionError: If E_total is not divisible by n_heads
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % n_heads == 0, "Embedding dim is not divisible by n_heads"
        self.E_head = E_total // n_heads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=True,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, n_heads, E_head) -> (N, n_heads, L_t, E_head)
        query = query.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        key = key.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        value = value.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, n_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Consists of multi-head self-attention followed by a position-wise
    feedforward network, with residual connections and layer normalization.

    Copied from pytorch:
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        dtype: torch.dtype,
        device: str = "cpu",
    ):
        """
        Initialize the TransformerEncoderLayer module.

        Args:
            d_model (int): Model dimension (input/output size)
            n_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network hidden layer
            dropout (float): Dropout probability
            activation (nn.Module): Activation function for feedforward network
            layer_norm_eps (float): Epsilon for layer normalization
            norm_first (bool): If True, apply layer norm before attention/feedforward
            bias (bool): Whether to use bias in linear layers
            dtype (torch.dtype): Data type for parameters
            device (str): Device to place the model on (default: "cpu")
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            n_heads,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask, is_causal):
        """
        Self-attention block with dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len)
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        """Feedforward block with dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=True):
        """
        Forward pass through the encoder layer.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len)
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            out_1 = self._sa_block(x, src_mask, is_causal)
            if out_1.dim() == 4:
                # Required for rollout transformer
                out_2 = out_1 + x.unsqueeze(1).expand_as(out_1)
            else:
                # Standard transformers
                out_2 = out_1 + x
            x = self.norm1(out_2)
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers.

    Applies multiple encoder layers sequentially with optional final normalization.

    Copied from pytorch:
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module],
        dtype: torch.dtype,
        device: str = "cpu",
    ):
        """
        Initialize the TransformerEncoder module.

        Args:
            encoder_layer (nn.Module): Single encoder layer to clone
            num_layers (int): Number of encoder layers
            norm (nn.Module, optional): Final layer normalization
            dtype (torch.dtype): Data type for parameters
            device (str): Device to place the model on (default: "cpu")
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal=True
    ):
        """
        Forward pass through all encoder layers.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = src
        for mod in self.layers:
            output = mod(output, mask, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    """
    Standard transformer encoder for sequence modeling.

    Implements input embedding, positional encoding, and stacked encoder layers
    for sequence-to-sequence transformation.

    Copied from pytorch:
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the Transformer module.

        Args:
            d_model (int): Input dimension
            n_heads (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout probability
            activation (nn.Module): Activation function for feedforward layers
            layer_norm_eps (float): Epsilon for layer normalization
            norm_first (bool): Whether to apply layer norm before attention
            bias (bool): Whether to use bias in linear layers
            input_length (int): Maximum input sequence length
            hidden_size (int): Hidden dimension size
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()

        self.input_embedding = nn.Linear(d_model, hidden_size, bias=bias, device=device)

        encoder_layer = TransformerEncoderLayer(
            hidden_size,  # Fix: Use d_model instead of hidden_size
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            dtype=None,
            device=device,
        )

        encoder_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, bias=bias, device=device
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
            dtype=None,
            device=device,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size,
            sequence_length=input_length + 10,  # Provide some buffer
            dropout=dropout,
            device=device,
        )

    def forward(
        self,
        src,
        src_mask=None,
        is_causal=True,
    ):
        """
        Forward pass through the transformer encoder.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): Attention mask. Default: None
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full output (batch_size, 1, seq_len, hidden_size)
                - final_hidden_state: Last timestep (batch_size, 1, hidden_size)
                - output: Same as sequence_output (batch_size, 1, seq_len, hidden_size)
                - sindy_loss: None (no SINDy loss in vanilla transformer)
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

        transformer_output = einops.rearrange(transformer_output, "b s d -> b 1 s d")

        return {
            "sequence_output": transformer_output,
            "final_hidden_state": transformer_output[:, :, -1, :],
            "output": transformer_output,
            "sindy_loss": None,
        }
