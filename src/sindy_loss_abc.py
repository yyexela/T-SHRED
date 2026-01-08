import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
from pytorch_polynomial_features import PolynomialFeatures


class SINDyLoss(nn.Module):
    """Mixin for SINDy Loss"""

    def __init__(
        self,
        poly_order: int,
        dt: float,
        hidden_size: int,
        sindy_loss_threshold: float,
        *args,
        **kwargs,
    ):
        # Stupid? Maybe. Works? Yes. Forwards kwargs even though this class uses them.
        kwargs["hidden_size"] = hidden_size
        super().__init__(*args, **kwargs)

        self.poly_order = poly_order
        self.dt = dt
        self.hidden_size = hidden_size
        self.sindy_loss_threshold = sindy_loss_threshold

        pf = PolynomialFeatures(
            degree=poly_order, interaction_only=False, include_bias=False
        )
        self.pf = pf
        self.pf.fit(torch.randn(1, self.hidden_size))  # Necessary for output features
        self.library_dim = self.pf.n_output_features_

        # SINDy coefficients (learnable parameters)
        self.coefficients = nn.Parameter(
            torch.Tensor(self.library_dim, self.hidden_size)
        )
        nn.init.xavier_uniform_(
            self.coefficients, gain=0.0000000
        )  # Initialize with small values

        # Coefficient mask for thresholding (not learnable, used for sparsification)
        self.register_buffer(
            "coefficient_mask", torch.ones(self.library_dim, self.hidden_size)
        )

    def compute_sindy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate SINDy loss based on derivatives with torchdiffeq.
        Propogate forward all hidden states

        TODO: Ask Mars:
        - x is [batch, sequence_length, hidden_size]
        - Do we want to propogate all hidden states one step, or just the first one all the way?
          1) Hidden:   [x1   x2   x3   x4   x5]
             Calculate [x1+1 x2+1 x3+1 x4+1]
             Compare   [x2   x3   x4   x5]
          2) Hidden:   [x1   x2   x3   x4   x5]
             Calculate [x1+1 x1+2 x1+3 x1+4]
             Compare   [x2   x3   x4   x5]

        Args:
            x: Transformed sequence of shape (batch_size, sequence_length, hidden_size)

        Returns:
            torch.Tensor: SINDy regularization loss
        """
        batch_size, seq_len, hidden_size = x.shape

        if x.shape[0] < 3:
            return torch.tensor(0.0)

        x_0 = x[:, 0:-1, :]
        x_1 = x[:, 1:, :]

        x_0_poly = einops.rearrange(
            x_0, "batch seq_len hidden_size -> (batch seq_len) hidden_size"
        )
        library_theta = self.pf.fit_transform(x_0_poly)

        def f(t, y):
            y = y.reshape(library_theta.shape[0], library_theta.shape[1])
            y = y.T
            terms = self.coefficients.to(y.device)
            dy = terms @ y
            dy = dy.T
            return dy.flatten()

        t_eval = torch.linspace(0, 1, 2, device=library_theta.device).float()
        library_theta_flat = library_theta.flatten()
        rollout = odeint(f, library_theta_flat, t_eval, method="rk4")

        # Reshape update back to (forecast, batch_size, seq_len, hidden_size)
        rollout = einops.rearrange(
            rollout,
            "n (b s h) -> n b s h",
            n=t_eval.shape[0],
            b=batch_size,
            s=seq_len - 1,
            h=hidden_size,
        )

        effective_coefficients = self.coefficients * self.coefficient_mask

        step_loss = torch.mean(torch.square(rollout[1] - x_1))
        l2_loss = torch.mean(torch.square(effective_coefficients))
        total_loss = step_loss + 0.001 * l2_loss

        return total_loss

    def compute_sindy_loss_original(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate SINDy loss based on derivatives with a midpoint integration method.
        For each time step (t0 to t1), we integrate in two steps (t0 to t0.5, then t0.5 to t1).

        Args:
            x: Transformed sequence of shape (batch_size, sequence_length, hidden_size)

        Returns:
            torch.Tensor: SINDy regularization loss
        """
        batch_size, seq_len, hidden_size = x.shape

        if x.shape[0] < 3:
            return torch.tensor(0.0)

        # We need to compare: h_t -> h_{t+1} and h_{t+1} -> h_{t+2}
        h_t = x[:-2, :, :]  # (batch_size-2, 1, hidden_size)
        h_t_next = x[1:-1, :, :]  # (batch_size-2, 1, hidden_size)
        h_t_next2 = x[2:, :, :]  # (batch_size-2, 1, hidden_size)

        # Compute observed derivatives using explicit dt
        h_dot_observed = (h_t_next - h_t) / self.dt  # (batch_size-2, 1, hidden_size)

        # Reshape for SINDy library computation
        h_t_flat = h_t.reshape(-1, hidden_size)  # ((batch_size-2)*(1), hidden_size)

        # Compute SINDy library features for h_t
        library_theta_t = self.pf.fit_transform(h_t_flat)

        # Apply coefficient mask (for sparsity)
        effective_coefficients = self.coefficients * self.coefficient_mask

        # Calculate SINDy derivative predictions for h_t
        h_dot_pred = library_theta_t @ effective_coefficients
        h_dot_pred = h_dot_pred.reshape(batch_size - 2, 1, hidden_size)

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
        h_dot_mid_pred = h_dot_mid_pred.reshape(batch_size - 2, 1, hidden_size)

        # Step 3: Second half-step - use midpoint derivatives to predict h_{t+1}
        h_t_next_pred = (
            h_t_mid_pred + h_dot_mid_pred * half_dt
        )  # Use full dt but with midpoint derivatives

        # Step 4: Compute prediction loss for first time step
        first_step_loss = torch.mean((h_t_next_pred - h_t_next) ** 2)

        # ---------- Repeat the process for the next time step (t+1 to t+2) ----------

        # Step 5: Compute derivatives at predicted h_{t+1}
        h_t_next_flat = h_t_next_pred.reshape(-1, hidden_size)
        library_theta_next = self.pf.fit_transform(h_t_next_flat)
        h_dot_next_pred = library_theta_next @ effective_coefficients
        h_dot_next_pred = h_dot_next_pred.reshape(batch_size - 2, 1, hidden_size)

        # Step 6: First half-step from h_{t+1} - predict h_{t+1.5}
        h_t_next_mid_pred = h_t_next_pred + h_dot_next_pred * half_dt

        # Step 7: Compute derivatives at the midpoint h_{t+1.5}
        h_t_next_mid_flat = h_t_next_mid_pred.reshape(-1, hidden_size)
        library_theta_next_mid = self.pf.fit_transform(h_t_next_mid_flat)
        h_dot_next_mid_pred = library_theta_next_mid @ effective_coefficients
        h_dot_next_mid_pred = h_dot_next_mid_pred.reshape(
            batch_size - 2, 1, hidden_size
        )

        # Step 8: Second half-step - use midpoint derivatives to predict h_{t+2}
        h_t_next2_pred = (
            h_t_next_mid_pred + h_dot_next_mid_pred * half_dt
        )  # Use full dt but with midpoint derivatives

        # Step 9: Compute prediction loss for second time step
        second_step_loss = torch.mean((h_t_next2_pred - h_t_next2) ** 2)

        # Add L1 regularization for sparsity
        l2_loss = torch.mean(torch.square(effective_coefficients))

        # Combine all losses
        total_loss = (
            derivative_loss + first_step_loss + second_step_loss + 0.001 * l2_loss
        )

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
