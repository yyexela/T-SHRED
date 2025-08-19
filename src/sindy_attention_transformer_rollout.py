import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
from vanilla_transformer import Transformer
from positional_encoding import PositionalEncoding
from pytorch_polynomial_features import PolynomialFeatures

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class MultiHeadSindyAttentionRollout(nn.Module):
    """
    Computes multi-head attention with latent space rollout. Supports nested or padded tensors.

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
        forecast_length: int,
        dropout: float,
        bias: bool,
        poly_order: int,
        dtype: torch.dtype,
        device='cpu',
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Class variables
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        self.bias = bias
        self.poly_order = poly_order
        self.forecast_length = forecast_length

        # Create projection matrices (Q K V)
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)

        # Create output projection matrix
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)

        # Check if embedding dim is divisible by nheads
        if E_total % nheads != 0:
            raise ValueError("Embedding dim is not divisible by nheads")
        self.E_head = E_total // nheads

        # Create SINDy Attention library
        self.pf = PolynomialFeatures(degree=poly_order)
        self.pf.fit(torch.randn(1, self.E_head)) # Necessary for output features
        self.library_dim = self.pf.n_output_features_
        self.coefficients = nn.ParameterList([torch.Tensor(self.library_dim, self.E_head) for _ in range(nheads)])
        self.library_terms = self.pf.get_feature_names_out()

        # Initialize SINDy Attention coefficients
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
        # coeffs: n_terms x hidden_dim
        # library_Theta: (batch x window len) x n_terms
        sindy_attn_output = []
        for i in range(self.nheads):
            # Extract head
            head = attn_output[:,i,:,:]
            # Reshape src for sindy_library (batch_size * seq_len, hidden_size/nheads + 1 for linear)
            head = einops.rearrange(head, 'b s h -> (b s) h', b=attn_output.shape[0], s=attn_output.shape[2],  h=self.E_head)
            # Calculate SINDy library features
            library_Theta = self.pf.fit_transform(head)
            # Calculate SINDy update (use masked coefficients)
            # effective_coefficients = self.coefficients * self.coefficient_mask.to(self.coefficients.device) # Ensure mask is on correct device
            ############################## Simplified SINDy update (without mask) #############################
            # Forecast n steps
            # > library_Theta: (batch x window len) x n_terms
            # > coefficients: n_heads x ((library terms + 1 (for linear) terms) x library_terms equations)
            # > Initial condition is from library_Theta, propogate forward n steps
            
            def f(t, y):
                y = y.reshape(library_Theta.shape[0], library_Theta.shape[1] - 1)
                # add linear term back
                y = torch.cat([torch.ones(y.shape[0], 1, device=y.device), y], dim=1)
                y = y.T
                dy = self.coefficients[i].T @ y
                dy = dy.T
                return dy.flatten()
            
            t_eval = torch.arange(1, self.forecast_length+1, 1, device=library_Theta.device).float()
            library_Theta_flat = library_Theta[:,1:].flatten() # don't include the linear term when passing into odeint
            rollout = odeint(f, library_Theta_flat, t_eval, method='rk4')
            rollout = rollout.reshape(self.forecast_length, library_Theta.shape[0], library_Theta.shape[1] - 1)

            # Reshape update back to (batch_size, seq_len, hidden_size)
            rollout = einops.rearrange(rollout, 'n (b s) h -> b n s h', n=self.forecast_length, b=attn_output.shape[0], s=attn_output.shape[2],  h=self.E_head)
            sindy_attn_output.append(rollout)
        sindy_attn_output = torch.stack(sindy_attn_output, dim=2)

        attn_output = sindy_attn_output.transpose(2, 3).flatten(-2) # 2 x 20 x 12

        # Step 5. Apply output projection (ff network)
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class SindyAttentionTransformerRollout(Transformer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        forecast_length: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        poly_order: int,
        device: str = 'cpu',
    ):
        super().__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, bias=bias, input_length=input_length, hidden_size=hidden_size, device=device)

        for layer in self.encoder.layers:
            layer.self_attn = MultiHeadSindyAttentionRollout(
                hidden_size,
                hidden_size,
                hidden_size,
                hidden_size,
                nhead,
                forecast_length=forecast_length,
                dropout=dropout,
                bias=bias,
                poly_order=poly_order,
                device=device,
                dtype=None
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
            "sequence_output": transformer_output, # [rollout, batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, :, -1, :], # Last timestep [batch_size, rollout, d_model]
            "sindy_loss": None
        }
    