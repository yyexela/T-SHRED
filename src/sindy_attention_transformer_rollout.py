import copy
import torch
import einops
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torchdiffeq import odeint
from positional_encoding import PositionalEncoding

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
        forecast_length=1,
        dropout: float = 0.0,
        bias=True,
        poly_order=2,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self.forecast_length = forecast_length
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
        self.library_dim = calculate_library_dim(self.E_head, poly_order, include_sine) # (hidden_dim / n_heads) + 1 for linear
        self.coefficients = nn.ParameterList([torch.Tensor(self.library_dim, self.E_head) for _ in range(nheads)]) # n_heads x library_dim x (hidden_dim / n_heads)
        self.initial_conditions = nn.Parameter(torch.Tensor(nheads, self.E_head)) # n_heads x (hidden_dim / n_heads)
        self.library_terms = sindy_library_terms(self.E_head, poly_order, include_sine)
        for i in range(nheads):
            nn.init.xavier_uniform_(self.coefficients[i])
        nn.init.xavier_uniform_(self.initial_conditions)

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
            library_theta = sindy_library_torch(head, self.E_head, self.poly_order, self.include_sine)
            # Calculate SINDy update (use masked coefficients)
            # effective_coefficients = self.coefficients * self.coefficient_mask.to(self.coefficients.device) # Ensure mask is on correct device
            ############################## Simplified SINDy update (without mask) #############################
            # Forecast n steps
            # > library_Theta: (batch x window len) x n_terms
            # > coefficients: n_heads x ((library terms + 1 (for linear) terms) x library_terms equations)
            # > Initial condition is from library_Theta, propogate forward n steps
            
            def f(t, y):
                y = y.reshape(library_theta.shape[0], library_theta.shape[1] - 1)
                # add linear term back
                y = torch.cat([torch.ones(y.shape[0], 1, device=y.device), y], dim=1)
                y = y.T
                dy = self.coefficients[i].T @ y
                dy = dy.T
                return dy.flatten()
            
            t_eval = torch.arange(1, self.forecast_length+1, 1, device=library_theta.device).float()
            library_theta_flat = library_theta[:,1:].flatten() # don't include the linear term when passing into odeint
            rollout = odeint(f, library_theta_flat, t_eval, method='rk4')
            rollout = rollout.reshape(self.forecast_length, library_theta.shape[0], library_theta.shape[1] - 1)

            #sindy_update = library_Theta @ self.coefficients[i]
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
class TransformerSindyEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        forecast_length=1,
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
            forecast_length=forecast_length,
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
            out_1 = self._sa_block(x, src_mask, is_causal)
            out_2 = out_1 + x.unsqueeze(1).expand_as(out_1)
            x = self.norm1(out_2)
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
class SindyAttentionTransformerRollout(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        forecast_length=1,
        num_encoder_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
        bias=True,
        input_length=10,
        hidden_size=10,
        poly_order=2,
        include_sine=False,
        sindy_loss=False,
        dt=1.0,
        device='cpu',
    ):
        super().__init__()

        self.sindy_loss = sindy_loss
        self.dt = dt
        self.poly_order = poly_order
        self.include_sine = include_sine

        if num_encoder_layers > 1:
            raise ValueError("num_encoder_layers must be 1 for SindyAttentionTransformerRollout")

        encoder_layer = TransformerSindyEncoderLayer(
            hidden_size,
            nhead,
            forecast_length,
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
            sequence_length=input_length + 10, # Provide some buffer
            dropout=dropout
        )

        self.input_embedding = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size, # GRU output matches d_model
            num_layers=2,                 # Example: 2 GRU layers for embedding
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0.0 # Dropout between GRU layers
        )
        
        if self.sindy_loss:
            # SINDy components
            self.library_dim = calculate_library_dim(hidden_size, poly_order, include_sine)
            
            # SINDy coefficients (learnable parameters)
            self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, hidden_size))
            nn.init.xavier_uniform_(self.coefficients, gain=0.0000000)  # Initialize with small values
            
            # Coefficient mask for thresholding (not learnable, used for sparsification)
            self.register_buffer('coefficient_mask', torch.ones(self.library_dim, hidden_size))

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

        sindy_loss = self.compute_sindy_loss(transformer_output) if self.sindy_loss else None

        return {
            "sequence_output": transformer_output, # [rollout, batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, :, -1, :], # Last timestep [batch_size, rollout, d_model]
            "sindy_loss": sindy_loss
        }
    
    def compute_sindy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate SINDy loss based on derivatives with a midpoint integration method.
        For each time step (t0 to t1), we integrate in two steps (t0 to t0.5, then t0.5 to t1).
        
        Args:
            x: Transformed sequence of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            torch.Tensor: SINDy regularization loss
        """
        rollout_size, batch_size, seq_len, hidden_size = x.shape
        
        # We need to compare: h_t -> h_{t+1} and h_{t+1} -> h_{t+2}
        h_t = x[:, :, :-2, :]          # (rollout_size, batch_size, seq_len-2, hidden_size)
        h_t_next = x[:, :, 1:-1, :]    # (rollout_size, batch_size, seq_len-2, hidden_size)
        h_t_next2 = x[:, :, 2:, :]     # (rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Compute observed derivatives using explicit dt
        h_dot_observed = (h_t_next - h_t) / self.dt  # (rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Reshape for SINDy library computation
        h_t_flat = h_t.reshape(-1, hidden_size)  # (rollout_size*batch_size*(seq_len-2), hidden_size)
        
        # Compute SINDy library features for h_t
        library_theta_t = sindy_library_torch(h_t_flat, hidden_size, self.poly_order, self.include_sine)
        
        # Apply coefficient mask (for sparsity)
        effective_coefficients = self.coefficients * self.coefficient_mask
        
        # Calculate SINDy derivative predictions for h_t
        h_dot_pred = library_theta_t @ effective_coefficients
        h_dot_pred = h_dot_pred.reshape(rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Calculate loss between SINDy derivative predictions and observed derivatives
        derivative_loss = torch.mean((h_dot_pred - h_dot_observed) ** 2)
        
        # ---------- Two-step integration within one time step (midpoint method) ----------
        
        # Step 1: First half-step - predict h_{t+0.5} using Euler forward
        half_dt = self.dt / 2.0
        h_t_mid_pred = h_t + h_dot_pred * half_dt
        
        # Step 2: Compute derivatives at the midpoint h_{t+0.5}
        h_t_mid_flat = h_t_mid_pred.reshape(-1, hidden_size)
        library_theta_mid = sindy_library_torch(h_t_mid_flat, hidden_size, self.poly_order, self.include_sine)
        h_dot_mid_pred = library_theta_mid @ effective_coefficients
        h_dot_mid_pred = h_dot_mid_pred.reshape(rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Step 3: Second half-step - use midpoint derivatives to predict h_{t+1}
        h_t_next_pred = h_t_mid_pred + h_dot_mid_pred * half_dt  # Use full dt but with midpoint derivatives
        
        # Step 4: Compute prediction loss for first time step
        first_step_loss = torch.mean((h_t_next_pred - h_t_next) ** 2)
        
        # ---------- Repeat the process for the next time step (t+1 to t+2) ----------
        
        # Step 5: Compute derivatives at predicted h_{t+1}
        h_t_next_flat = h_t_next_pred.reshape(-1, hidden_size)
        library_theta_next = sindy_library_torch(h_t_next_flat, hidden_size, self.poly_order, self.include_sine)
        h_dot_next_pred = library_theta_next @ effective_coefficients
        h_dot_next_pred = h_dot_next_pred.reshape(rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Step 6: First half-step from h_{t+1} - predict h_{t+1.5}
        h_t_next_mid_pred = h_t_next_pred + h_dot_next_pred * half_dt
        
        # Step 7: Compute derivatives at the midpoint h_{t+1.5}
        h_t_next_mid_flat = h_t_next_mid_pred.reshape(-1, hidden_size)
        library_theta_next_mid = sindy_library_torch(h_t_next_mid_flat, hidden_size, self.poly_order, self.include_sine)
        h_dot_next_mid_pred = library_theta_next_mid @ effective_coefficients
        h_dot_next_mid_pred = h_dot_next_mid_pred.reshape(rollout_size, batch_size, seq_len-2, hidden_size)
        
        # Step 8: Second half-step - use midpoint derivatives to predict h_{t+2}
        h_t_next2_pred = h_t_next_mid_pred + h_dot_next_mid_pred * half_dt  # Use full dt but with midpoint derivatives
        
        # Step 9: Compute prediction loss for second time step
        second_step_loss = torch.mean((h_t_next2_pred - h_t_next2) ** 2)
        
        # Add L1 regularization for sparsity
        l2_loss = torch.mean(torch.square(effective_coefficients))
        
        # Combine all losses
        total_loss = derivative_loss + first_step_loss + second_step_loss + 0.001*l2_loss

        return total_loss

    def get_SINDy_coefficients_sum(self):
        """
        Sum of all SINDy coefficients in all heads of all layers.
        """
        with torch.no_grad():
            sindy_sum = 0.
            for i, layer in enumerate(self.encoder.layers):
                for i in range(layer.self_attn.nheads):
                    sindy_sum += torch.sqrt((torch.abs(layer.self_attn.coefficients[i].data)**2).sum())
        return sindy_sum

    def threshold_all_layers(self, threshold):
        """
        Threshold all SINDy coefficients in all heads of all layers.
        """
        for i, layer in enumerate(self.encoder.layers):
            print(f"Layer {i}")
            with torch.no_grad():
                for i in range(layer.self_attn.nheads):
                    mask = torch.abs(layer.self_attn.coefficients[i].data) > threshold
                    layer.self_attn.coefficients[i].data *= mask
                    print(f"SindyAttentionTransformer: Applied threshold {threshold} to head {i}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}")
            print()
