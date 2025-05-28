import torch
import torch.nn as nn
from helpers import calculate_library_dim, sindy_library_torch

class SINDyLossGRU(nn.Module):
    def __init__(self, input_size:int = 3,
                 hidden_size:int = 64,
                 num_layers:int = 2,
                 dropout:float = 0.1,
                 poly_order: int = 2,
                 include_sine: bool = False,
                 sindy_loss_threshold: float = 0.05,      # Threshold for SINDy coefficient sparsification
                 dt: float = 1.0,                    # Time step for SINDy derivatives
                 device: str = 'cpu'
                ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.sindy_loss_threshold = sindy_loss_threshold
        self.dt = dt
        self.device = device

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # SINDy components
        self.library_dim = calculate_library_dim(self.hidden_size, self.poly_order, self.include_sine)

        # SINDy coefficients (learnable parameters)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.hidden_size))
        nn.init.xavier_uniform_(self.coefficients, gain=0.0000000)  # Initialize with small values

        # Coefficient mask for thresholding (not learnable, used for sparsification)
        self.register_buffer('coefficient_mask', torch.ones(self.library_dim, self.hidden_size))

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, h_out = self.gru(x, h_0)

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size),
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

class SINDyLossLSTM(nn.Module):
    def __init__(self, input_size:int = 3,
                 hidden_size:int = 64,
                 num_layers:int = 2,
                 dropout:float = 0.1,
                 poly_order: int = 2,
                 include_sine: bool = False,
                 sindy_loss_threshold: float = 0.05,      # Threshold for SINDy coefficient sparsification
                 dt: float = 1.0,                    # Time step for SINDy derivatives
                 device: str = 'cpu'
                ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.sindy_loss_threshold = sindy_loss_threshold
        self.dt = dt
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # SINDy components
        self.library_dim = calculate_library_dim(self.hidden_size, self.poly_order, self.include_sine)

        # SINDy coefficients (learnable parameters)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.hidden_size))
        nn.init.xavier_uniform_(self.coefficients, gain=0.0000000)  # Initialize with small values

        # Coefficient mask for thresholding (not learnable, used for sparsification)
        self.register_buffer('coefficient_mask', torch.ones(self.library_dim, self.hidden_size))

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        sindy_loss = self.compute_sindy_loss(out)

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size),
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