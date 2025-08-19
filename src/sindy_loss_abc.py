import torch
import torch.nn as nn
from pytorch_polynomial_features import PolynomialFeatures

class SINDyLoss(nn.Module):
    """Mixin for SINDy Loss"""
    def __init__(self, poly_order, dt, hidden_size, *args, **kwargs):
        kwargs['hidden_size'] = hidden_size # Stupid? Maybe. Works? Yes.
        super().__init__(*args, **kwargs)
        self.poly_order = poly_order
        self.dt = dt
        self.hidden_size = hidden_size

        pf = PolynomialFeatures(degree=poly_order,
                                interaction_only=False,
                                include_bias=True)
        self.pf = pf
        self.pf.fit(torch.randn(1, self.hidden_size)) # Necessary for output features
        self.library_dim = self.pf.n_output_features_

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
        library_theta_t = self.pf.fit_transform(h_t_flat)
        
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
        library_theta_mid = self.pf.fit_transform(h_t_mid_flat)
        h_dot_mid_pred = library_theta_mid @ effective_coefficients
        h_dot_mid_pred = h_dot_mid_pred.reshape(batch_size, seq_len-2, hidden_size)
        
        # Step 3: Second half-step - use midpoint derivatives to predict h_{t+1}
        h_t_next_pred = h_t_mid_pred + h_dot_mid_pred * half_dt  # Use full dt but with midpoint derivatives
        
        # Step 4: Compute prediction loss for first time step
        first_step_loss = torch.mean((h_t_next_pred - h_t_next) ** 2)
        
        # ---------- Repeat the process for the next time step (t+1 to t+2) ----------
        
        # Step 5: Compute derivatives at predicted h_{t+1}
        h_t_next_flat = h_t_next_pred.reshape(-1, hidden_size)
        library_theta_next = self.pf.fit_transform(h_t_next_flat)
        h_dot_next_pred = library_theta_next @ effective_coefficients
        h_dot_next_pred = h_dot_next_pred.reshape(batch_size, seq_len-2, hidden_size)
        
        # Step 6: First half-step from h_{t+1} - predict h_{t+1.5}
        h_t_next_mid_pred = h_t_next_pred + h_dot_next_pred * half_dt
        
        # Step 7: Compute derivatives at the midpoint h_{t+1.5}
        h_t_next_mid_flat = h_t_next_mid_pred.reshape(-1, hidden_size)
        library_theta_next_mid = self.pf.fit_transform(h_t_next_mid_flat)
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
