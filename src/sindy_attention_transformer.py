import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
from sindy_layer import SindyLayer
from vanilla_transformer import Transformer


# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class MultiHeadSindyAttention(nn.Module):
    """
    Computes multi-head attention with latent space rollout. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // n_heads
        n_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        n_heads: int,
        forecast_length: int,
        dropout: float,
        strict_symmetry: bool,
        bias: bool,
        dtype: torch.dtype,
        device="cpu",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Class variables
        self.n_heads = n_heads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        self.forecast_length = forecast_length
        self.device = device
        self.strict_symmetry = strict_symmetry

        # Create projection matrices (Q K V)
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)

        # Create output projection matrix
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)

        # Check if embedding dim is divisible by n_heads
        if E_total % n_heads != 0:
            raise ValueError("Embedding dim is not divisible by n_heads")
        self.E_head = E_total // n_heads

        # Initialize SINDy Attention layers
        self.sindy_layers = nn.ModuleList(
            [
                SindyLayer(
                    d_model=self.E_head,
                    forecast_length=self.forecast_length,
                    device=self.device,
                    strict_symmetry=self.strict_symmetry,
                )
                for _ in range(self.n_heads)
            ]
        )

    def matrix_from_params(self, head_idx):
        terms = torch.zeros(
            self.library_dim,
            self.library_dim,
            device=self.coefficients[head_idx].device,
        )
        self.tril_indices = self.tril_indices.to(terms.device)
        terms[self.tril_indices[0], self.tril_indices[1]] = self.coefficients[head_idx]
        terms = terms + terms.t() - torch.diag(terms.diag())
        terms = torch.tensor(1j) * terms
        return terms

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
        )  # 2 x 6 x 20 x 2
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)

        batch_size, _, seq_len, _ = attn_output.shape

        # Step 4. Per-head pysindy
        # coeffs: n_terms x hidden_dim
        # library_Theta: (batch x window len) x n_terms
        sindy_attn_output = []
        for i in range(self.n_heads):
            # Extract head values
            head = attn_output[:, i, :, :]
            # Reshape for input to sindy layer
            head = einops.rearrange(head, "b s h -> (b s) h")
            # Pass through sindy layer
            rollout = self.sindy_layers[i](head)

            # Reshape update back to (forecast, batch_size, seq_len, hidden_size)
            rollout = einops.rearrange(
                rollout,
                "(b s) n h -> b n s h",
                n=self.forecast_length,
                b=batch_size,
                s=seq_len,
                h=self.E_head,
            )
            sindy_attn_output.append(rollout)
        sindy_attn_output = torch.stack(sindy_attn_output, dim=2)

        attn_output = sindy_attn_output.transpose(2, 3).flatten(-2)  # 2 x 20 x 12

        # Step 5. Apply output projection (ff network)
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


# Copied from pytorch:
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class SindyAttentionTransformer(Transformer):
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
        strict_symmetry: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        device: str = "cpu",
    ):
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
            device=device,
        )

        self.encoder.layers[-1].self_attn = MultiHeadSindyAttention(
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            n_heads,
            forecast_length=forecast_length,
            dropout=dropout,
            bias=bias,
            strict_symmetry=strict_symmetry,
            device=device,
            dtype=None,
        )

        self.n_heads = n_heads

    def print_sindy_layer_coefficients(self):
        # coefficients: n_heads x ((library terms + 1 (for linear) terms) x library_terms equations)
        for j in range(self.n_heads):
            print(f"Head {j}:")
            coefficients = (
                self.encoder.layers[-1]
                .self_attn.sindy_layers[j]
                .get_dense_sindy_coefficients()
            )
            library = (
                self.encoder.layers[-1]
                .self_attn.sindy_layers[j]
                .pf.get_feature_names_out()
            )
            for k in range(coefficients.shape[1]):
                print(f"Hidden layer {k}:")
                output_str = ""
                for l in range(coefficients.shape[0]):
                    output_str += (
                        f"{coefficients[l][k].item():.3f} \\cdot {library[l]} + "
                    )
                print(output_str[:-3])
            print()

    def get_sindy_layer_coefficients_eigenvalues(self):
        with torch.no_grad():
            eigvs_l = []
            for i in range(self.n_heads):
                eigvs_l.append(
                    self.encoder.layers[-1].self_attn.sindy_layers[i].get_eigenvalues()
                )
            return eigvs_l

    def get_sindy_layer_coefficients_sum(self):
        """
        Sum of all SINDy coefficients in all heads of all layers.
        """
        with torch.no_grad():
            sindy_sum = 0.0
            layer = self.encoder.layers[-1]
            for i in range(layer.self_attn.n_heads):
                sindy_sum += torch.sqrt(
                    (torch.abs(layer.self_attn.coefficients[i].data) ** 2).sum()
                )
        return sindy_sum

    def set_forecast_length(self, forecast_length):
        # Set forecast length to expected plot length
        self.encoder.layers[-1].self_attn.forecast_length = forecast_length
        for i in range(self.n_heads):
            self.encoder.layers[-1].self_attn.sindy_layers[
                i
            ].forecast_length = forecast_length

    def threshold_sindy_layer_coefficients(self, threshold, verbose=False):
        """
        Threshold all SINDy coefficients in all heads of all layers.
        """
        layer = self.encoder.layers[-1]
        with torch.no_grad():
            for i in range(layer.self_attn.n_heads):
                mask = (
                    torch.abs(
                        layer.self_attn.sindy_layers[i].get_raw_sindy_coefficients()
                    )
                    > threshold
                )
                layer.self_attn.sindy_layers[i].set_raw_sindy_coefficients(
                    layer.self_attn.sindy_layers[i].get_raw_sindy_coefficients() * mask
                )
                if verbose:
                    print(
                        f"SindyAttentionTransformer: Applied threshold {threshold} to head {i}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}"
                    )
        if verbose:
            print()

    def forward(
        self,
        src,
        src_mask=None,
        is_causal=True,
    ):
        # Embed input
        x_embedded = self.input_embedding(src)

        # Apply positional encoding
        x_pos_encoded = self.pos_encoder(
            x_embedded
        )  # Shape: (batch_size, seq_len, d_model)

        transformer_output = self.encoder(
            x_pos_encoded,
            mask=src_mask,
            is_causal=is_causal,
        )

        return {
            "sequence_output": transformer_output,  # [forecast_length, batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[
                :, :, -1, :
            ],  # Last timestep [batch_size, forecast_length, d_model]
            "output": transformer_output,  # [batch_size, forecast_length, sequence_length, d_model]
            "sindy_loss": None,
        }
