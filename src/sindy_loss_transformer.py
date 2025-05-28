import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from helpers import calculate_library_dim, sindy_library_torch

class SINDyLossTransformer(nn.Module):
    """
    Transformer model with additional SINDy loss for learning sparse dynamics.
    This model implements a standard transformer encoder with an additional SINDy component 
    that is used to regularize the latent dynamics.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        hidden_size: int = 64,
        window_length: int = 10,
        num_encoder_layers: int = 3,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        poly_order: int = 2,
        include_sine: bool = False,
        device: str = 'cpu',
        sindy_loss_threshold: float = 0.05,      # Threshold for SINDy coefficient sparsification
        dt: float = 1.0,                    # Time step for SINDy derivatives
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.num_encoder_layers = num_encoder_layers
        self.window_length = window_length
        self.device = device
        self.sindy_loss_threshold = sindy_loss_threshold
        self.dt = dt  # Time step for Euler integration
        
        # Create the standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
            bias=bias,
            device=device
        )
        
        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=bias, device=device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_size,
            sequence_length=window_length + 10,  # Provide some buffer
            dropout=dropout
        )
        
        # Input embedding GRU
        self.input_embedding = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,  # 2 GRU layers for embedding
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0.0
        )
        
        # SINDy components
        self.library_dim = calculate_library_dim(hidden_size, poly_order, include_sine)
        
        # SINDy coefficients (learnable parameters)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, hidden_size))
        nn.init.xavier_uniform_(self.coefficients, gain=0.0000000)  # Initialize with small values
        
        # Coefficient mask for thresholding (not learnable, used for sparsification)
        self.register_buffer('coefficient_mask', torch.ones(self.library_dim, hidden_size))
    
    def forward(self, src: torch.Tensor) -> dict:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, d_model)
        Returns:
            Dictionary containing:
                - sequence_output: Output tensor of shape (batch_size, sequence_length, hidden_size)
                - final_hidden_state: Last timestep hidden state (batch_size, hidden_size)
                - sindy_loss: SINDy regularization loss if training (or None if not)
        """
        # Apply input embedding GRU
        x_embedded, _ = self.input_embedding(src)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Apply positional encoding
        x_pos = self.pos_encoder(x_embedded)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Apply transformer encoder
        x_transformed = self.transformer_encoder(x_pos)  # Shape: (batch_size, seq_len, hidden_size)

        # Calculate SINDy loss
        sindy_loss = self.compute_sindy_loss(x_transformed)
        
        return {
            "sequence_output": x_transformed,  # [batch_size, sequence_length, hidden_size]
            "final_hidden_state": x_transformed[:, -1, :],  # last timestep [batch_size, hidden_size]
            "sindy_loss": sindy_loss  # SINDy regularization loss
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
        batch_size, seq_len, hidden_size = x.shape
        
        # We need to compare: h_t -> h_{t+1} and h_{t+1} -> h_{t+2}
        h_t = x[:, :-2, :]          # (batch_size, seq_len-2, hidden_size)
        h_t_next = x[:, 1:-1, :]    # (batch_size, seq_len-2, hidden_size)
        h_t_next2 = x[:, 2:, :]     # (batch_size, seq_len-2, hidden_size)
        
        # Compute observed derivatives using explicit dt
        h_dot_observed = (h_t_next - h_t) / self.dt  # (batch_size, seq_len-2, hidden_size)
        
        # Reshape for SINDy library computation
        h_t_flat = h_t.reshape(-1, hidden_size)  # (batch_size*(seq_len-2), hidden_size)
        
        # Compute SINDy library features for h_t
        library_theta_t = sindy_library_torch(h_t_flat, hidden_size, self.poly_order, self.include_sine)
        
        # Apply coefficient mask (for sparsity)
        effective_coefficients = self.coefficients * self.coefficient_mask
        
        # Calculate SINDy derivative predictions for h_t
        h_dot_pred = library_theta_t @ effective_coefficients
        h_dot_pred = h_dot_pred.reshape(batch_size, seq_len-2, hidden_size)
        
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
        h_dot_mid_pred = h_dot_mid_pred.reshape(batch_size, seq_len-2, hidden_size)
        
        # Step 3: Second half-step - use midpoint derivatives to predict h_{t+1}
        h_t_next_pred = h_t_mid_pred + h_dot_mid_pred * half_dt  # Use full dt but with midpoint derivatives
        
        # Step 4: Compute prediction loss for first time step
        first_step_loss = torch.mean((h_t_next_pred - h_t_next) ** 2)
        
        # ---------- Repeat the process for the next time step (t+1 to t+2) ----------
        
        # Step 5: Compute derivatives at predicted h_{t+1}
        h_t_next_flat = h_t_next_pred.reshape(-1, hidden_size)
        library_theta_next = sindy_library_torch(h_t_next_flat, hidden_size, self.poly_order, self.include_sine)
        h_dot_next_pred = library_theta_next @ effective_coefficients
        h_dot_next_pred = h_dot_next_pred.reshape(batch_size, seq_len-2, hidden_size)
        
        # Step 6: First half-step from h_{t+1} - predict h_{t+1.5}
        h_t_next_mid_pred = h_t_next_pred + h_dot_next_pred * half_dt
        
        # Step 7: Compute derivatives at the midpoint h_{t+1.5}
        h_t_next_mid_flat = h_t_next_mid_pred.reshape(-1, hidden_size)
        library_theta_next_mid = sindy_library_torch(h_t_next_mid_flat, hidden_size, self.poly_order, self.include_sine)
        h_dot_next_mid_pred = library_theta_next_mid @ effective_coefficients
        h_dot_next_mid_pred = h_dot_next_mid_pred.reshape(batch_size, seq_len-2, hidden_size)
        
        # Step 8: Second half-step - use midpoint derivatives to predict h_{t+2}
        h_t_next2_pred = h_t_next_mid_pred + h_dot_next_mid_pred * half_dt  # Use full dt but with midpoint derivatives
        
        # Step 9: Compute prediction loss for second time step
        second_step_loss = torch.mean((h_t_next2_pred - h_t_next2) ** 2)
        
        # Add L1 regularization for sparsity
        l2_loss = torch.mean(torch.square(effective_coefficients))
        
        # Combine all losses
        total_loss = derivative_loss + first_step_loss + second_step_loss + 0.001*l2_loss

        return total_loss
    
    def thresholding(self, threshold=None):
        """
        Apply thresholding to SINDy coefficients to enforce sparsity.
        
        Args:
            threshold (float, optional): Threshold value. If None, uses the default threshold.
        """
        if threshold is None:
            threshold = self.sindy_loss_threshold
            
        with torch.no_grad():
            mask = torch.abs(self.coefficients.data) > threshold
            self.coefficients.data *= mask
            self.coefficient_mask.copy_(mask.float())
            
    def get_config(self):
        """
        Get model configuration parameters.
        
        Returns:
            dict: Dictionary of model parameters
        """
        return {
            "d_model": self.d_model,
            "hidden_size": self.hidden_size,
            "poly_order": self.poly_order,
            "include_sine": self.include_sine,
            "sindy_loss_threshold": self.sindy_loss_threshold,
            "library_dim": self.library_dim
        }
