"""
SINDy Attention Transformer module.

Implements multi-head attention with SINDy-based latent space rollouts
for learning interpretable dynamics in transformer architectures.
"""

import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
from sindy_layer import SindyLayer
from vanilla_transformer import Transformer


class MultiHeadSindyAttention(nn.Module):
    """
    Multi-head attention with SINDy-based latent space rollout.

    Replaces standard scaled dot-product attention output with ODE-based
    rollouts using learned SINDy dynamics. Each attention head has its
    own SINDy layer for independent dynamics learning.

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
        forecast_length: int,
        dropout: float,
        strict_symmetry: bool,
        bias: bool,
        dtype: torch.dtype,
        device="cpu",
    ):
        """
        Initialize the MultiHeadSindyAttention module.

        Args:
            E_q (int): Size of embedding dimension for query
            E_k (int): Size of embedding dimension for key
            E_v (int): Size of embedding dimension for value
            E_total (int): Total embedding dimension of combined heads post input projection.
                Each head has dimension E_total // n_heads
            n_heads (int): Number of attention heads
            forecast_length (int): Number of future timesteps to predict via ODE rollout
            dropout (float): Dropout probability for attention weights
            strict_symmetry (bool): Whether to enforce strict symmetry in SINDy coefficients
            bias (bool): Whether to add bias to input/output projections
            dtype (torch.dtype): Data type for parameters
            device (str): Device to place the model on (default: "cpu")

        Raises:
            ValueError: If E_total is not divisible by n_heads
        """
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
            attn_output (torch.Tensor): output of shape (batch_size, forecast_length, sequence_length, hidden_size)
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

            # Reshape update back to (batch_size, forecast_length, sequence_length, hidden_size)
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

        attn_output = sindy_attn_output.transpose(2, 3).flatten(-2)

        # Step 5. Apply output projection (ff network)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SindyAttentionTransformer(Transformer):
    """
    Transformer encoder with SINDy-based attention in the final layer.

    Extends the standard Transformer by replacing the attention mechanism
    in the last encoder layer with MultiHeadSindyAttention, enabling
    ODE-based latent space rollouts for multi-step forecasting.

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
        strict_symmetry: bool,
        bias: bool,
        input_length: int,
        hidden_size: int,
        device: str = "cpu",
    ):
        """
        Initialize the SindyAttentionTransformer module.

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
            strict_symmetry (bool): Whether to enforce strict symmetry in SINDy coefficients
            bias (bool): Whether to use bias in linear layers
            input_length (int): Length of input sequences
            hidden_size (int): Hidden dimension size
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
        """
        Print the SINDy layer coefficients for all attention heads.

        Displays the learned SINDy equations in human-readable format,
        showing the coefficient values and corresponding library terms
        for each hidden layer dimension.

        Returns:
            None
        """
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
        """
        Get eigenvalues of SINDy coefficient matrices for all attention heads.

        Returns:
            list: List of eigenvalue tensors, one per attention head
        """
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

        Returns:
            float: Sum of square roots of absolute SINDy coefficients
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
        """
        Set the forecast length for all SINDy attention layers.

        Args:
            forecast_length (int): Number of future timesteps to predict

        Returns:
            None
        """
        # Set forecast length to expected plot length
        self.encoder.layers[-1].self_attn.forecast_length = forecast_length
        for i in range(self.n_heads):
            self.encoder.layers[-1].self_attn.sindy_layers[
                i
            ].forecast_length = forecast_length

    def threshold_sindy_layer_coefficients(self, threshold, verbose=False):
        """
        Threshold all SINDy coefficients in all heads of all layers.

        Args:
            threshold (float): Threshold value for SINDy coefficients
            verbose (bool): Whether to print verbose output

        Returns:
            None
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
        """
        Forward pass through the SINDy attention transformer.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): Attention mask. Default: None
            is_causal (bool): Whether to apply causal masking. Default: True

        Returns:
            dict: Dictionary containing:
                - sequence_output: Full output (batch_size, forecast_length, seq_len, d_model)
                - final_hidden_state: Last timestep (batch_size, forecast_length, d_model)
                - output: Same as sequence_output (batch_size, forecast_length, sequence_length, hidden_size)
                - sindy_loss: None (SINDy loss not computed in this variant)
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

        return {
            "sequence_output": transformer_output,
            "final_hidden_state": transformer_output[:, :, -1, :],
            "output": transformer_output,
            "sindy_loss": None,
        }
