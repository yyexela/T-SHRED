import copy
import torch
import einops
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from positional_encoding import PositionalEncoding
from helpers import calculate_library_dim, sindy_library_torch

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class MultiHeadSindyAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        poly_order=2,
        include_sine=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
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
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = calculate_library_dim(self.E_head, poly_order, include_sine)
        self.coefficients = nn.ParameterList([torch.Tensor(self.library_dim, self.E_head) for _ in range(nheads)])
        for i in range(nheads):
            nn.init.xavier_uniform_(self.coefficients[i])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
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
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        ) # 2 x 6 x 20 x 2
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)

        # Step 4. Per-head pysindy
        sindy_attn_output = []
        for i in range(self.nheads):
            # Extract head
            head = attn_output[:,i,:,:]
            # Reshape src for sindy_library (batch_size * seq_len, hidden_size/nheads)
            head = einops.rearrange(head, 'b s h -> (b s) h', b=attn_output.shape[0], s=attn_output.shape[2],  h=self.E_head)
            # Calculate SINDy library features
            library_Theta = sindy_library_torch(head, self.E_head, self.poly_order, self.include_sine)
            # Calculate SINDy update (use masked coefficients)
            # effective_coefficients = self.coefficients * self.coefficient_mask.to(self.coefficients.device) # Ensure mask is on correct device
            ############################## Simplified SINDy update (without mask) #############################
            sindy_update = library_Theta @ self.coefficients[i]
            # Reshape update back to (batch_size, seq_len, hidden_size)
            sindy_update = einops.rearrange(sindy_update, '(b s) h -> b s h', b=attn_output.shape[0], s=attn_output.shape[2],  h=self.E_head)
            sindy_attn_output.append(sindy_update)
        sindy_attn_output = torch.stack(sindy_attn_output, dim=1)

        attn_output = sindy_attn_output.transpose(1, 2).flatten(-2) # 2 x 20 x 12

        # Step 5. Apply output projection (ff network)
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class TransformerSindyEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        poly_order=2,
        include_sine=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadSindyAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            poly_order=poly_order,
            include_sine=include_sine,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        

    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class TransformerSindyEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: "TransformerSindyEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal=False):
        output = src
        for mod in self.layers:
            output = mod(output, mask, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class SindyAttentionTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        bias=True,
        window_length=10,
        hidden_size=10,
        poly_order=2,
        include_sine=False,
        device='cpu',
    ):
        super().__init__()
        encoder_layer = TransformerSindyEncoderLayer(
            hidden_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            poly_order=poly_order,
            include_sine=include_sine,
        )

        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=bias, device=device)
        self.encoder = TransformerSindyEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size,
            sequence_length=window_length + 10, # Provide some buffer
            dropout=dropout
        )

        self.input_embedding = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size, # GRU output matches d_model
            num_layers=2,                 # Example: 2 GRU layers for embedding
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0.0 # Dropout between GRU layers
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_is_causal=False,
    ):
        x_embedded, _ = self.input_embedding(src) # Shape: (batch_size, seq_len, d_model)

        x_pos_encoded = self.pos_encoder(x_embedded) # Shape: (batch_size, seq_len, d_model)

        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=src_is_causal,
        )

        return {
            "sequence_output": transformer_output, # [batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, -1, :], # Last timestep [batch_size, d_model]
            "sindy_loss": None
        }
        
# We use this for exact parity with the PyTorch implementation, having the same init
# for every layer might not be necessary.
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])