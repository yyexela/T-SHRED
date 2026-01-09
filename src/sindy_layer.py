"""
SINDy Layer module.

Implements a differentiable SINDy (Sparse Identification of Nonlinear Dynamics)
layer for learning interpretable ODEs and performing arbitrary-length forecasting.
"""

import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
from pytorch_polynomial_features import PolynomialFeatures


class SindyLayer(nn.Module):
    """
    Differentiable SINDy layer for ODE-based forecasting.

    Learns sparse polynomial dynamics from data and uses ODE integration
    for arbitrary-length forecasting. Supports both strict symmetry
    (parameterized via lower triangle) and general coefficient matrices.

    Attributes:
        pf (PolynomialFeatures): Polynomial feature generator
        library_dim (int): Number of features in the polynomial library
        triangle_coefficients (nn.Parameter): Lower triangle coefficients (if strict_symmetry)
        sindy_coefficients (nn.Parameter): Full coefficient matrix (if not strict_symmetry)
    """

    def __init__(
        self,
        d_model: int,
        forecast_length: int,
        device: str = "cpu",
        strict_symmetry: bool = True,
        **kwargs,
    ):
        """
        Initialize the SindyLayer module.

        Args:
            d_model (int): Input/output dimension of the layer
            forecast_length (int): Number of future timesteps to predict
            device (str): Device to place the model on (default: "cpu")
            strict_symmetry (bool): If True, enforces symmetric coefficient matrix
                via lower triangle parameterization. Default: True
            **kwargs: Additional keyword arguments (ignored)
        """
        # Initialize parent class
        super().__init__()

        # Class variables
        self.d_model = d_model
        self.forecast_length = forecast_length
        self.device = device
        self.strict_symmetry = strict_symmetry

        # Polynomial features
        self.pf = PolynomialFeatures(degree=1, include_bias=False)
        self.pf.fit(torch.randn(1, self.d_model))
        self.library_dim = self.pf.n_output_features_

        # Initialize SINDy coefficients (SINDy library)
        # TODO: Initialization? Should be larger I think, or different per-layer for MOE?
        if self.strict_symmetry:
            # Symmetric parameters, builds a 1D list of parameters (lower triangle) that get converted to a dense symmetric matrix
            self.tril_indices = torch.tril_indices(self.library_dim, self.library_dim)
            num_params = (self.library_dim * (self.library_dim + 1)) // 2
            self.triangle_coefficients = nn.Parameter(torch.Tensor(num_params))
            nn.init.normal_(self.triangle_coefficients, mean=0.0, std=0.1)
        else:
            # General dense 1D matrix, initialized as a symmetric matrix
            num_params = self.library_dim * self.library_dim
            triangle_coefficients = torch.randn(num_params) * 0.1 + 0.0
            sindy_coefficients = self.dense_matrix_from_symmetric_params(
                triangle_coefficients
            )
            self.sindy_coefficients = nn.Parameter(sindy_coefficients)

    def get_dense_sindy_coefficients(self) -> torch.Tensor:
        """
        Convert symmetric parameters (1D list) to a dense matrix.
        Only used when `strict_symmetry` is True.

        Returns:
            torch.Tensor: Dense SINDy coefficients
        """
        if self.strict_symmetry:
            sindy_coefficients = self.dense_matrix_from_symmetric_params(
                self.triangle_coefficients
            )
            return sindy_coefficients
        else:
            return self.sindy_coefficients

    def get_raw_sindy_coefficients(self) -> torch.Tensor:
        """
        Get the raw SINDy coefficients (not converted to a dense matrix).

        Returns:
            torch.Tensor: Raw SINDy coefficients
        """
        if self.strict_symmetry:
            return self.triangle_coefficients
        else:
            return self.sindy_coefficients

    def set_raw_sindy_coefficients(self, coefficients: torch.Tensor):
        """
        Set the raw SINDy coefficients (not converted to a dense matrix).

        Args:
            coefficients (torch.Tensor): Raw SINDy coefficients

        Returns:
            None
        """
        if self.strict_symmetry:
            self.triangle_coefficients.data.copy_(coefficients)
        else:
            self.sindy_coefficients.data.copy_(coefficients)

    def dense_matrix_from_symmetric_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert symmetric parameters (1D list) to a dense matrix.
        Only used when `strict_symmetry` is True.

        Args:
            params (torch.Tensor): Symmetric parameters

        Returns:
            torch.Tensor: Dense SINDy coefficients
        """
        sindy_coefficients = torch.zeros(
            self.library_dim, self.library_dim, device=params.device
        )
        self.tril_indices = self.tril_indices.to(sindy_coefficients.device)
        sindy_coefficients[self.tril_indices[0], self.tril_indices[1]] = params
        sindy_coefficients = (
            sindy_coefficients
            + sindy_coefficients.t()
            - torch.diag(sindy_coefficients.diag())
        )
        return sindy_coefficients

    def get_eigenvalues(self) -> torch.Tensor:
        """
        Get the eigenvalues of the SINDy coefficients.

        Returns:
            torch.Tensor: Eigenvalues of the SINDy coefficients
        """
        sindy_coefficients = self.get_dense_sindy_coefficients()
        eigenvalues = torch.linalg.eigvals(
            sindy_coefficients.to(torch.cfloat) * torch.tensor(1j)
        )
        eigenvalues = eigenvalues.cpu()
        return eigenvalues

    def forward(self, x):
        """
        Forward pass: integrate learned ODE dynamics for forecasting.

        Transforms input through polynomial library, then integrates the
        learned ODE system using RK4 to produce multi-step forecasts.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size)

        Returns:
            torch.Tensor: Rollout predictions of shape (batch_size, forecast_length, hidden_size)
        """
        batch_size, hidden_size = x.shape
        sindy_coefficients = self.get_dense_sindy_coefficients()
        library_Theta = self.pf.fit_transform(x)

        def f(t, y):
            y = y.reshape(library_Theta.shape[0], library_Theta.shape[1])
            y = y.T
            terms = sindy_coefficients.to(y.device)
            terms = terms.to(torch.cfloat) * torch.tensor(1j)
            dy = terms @ y
            dy = dy.T
            return dy.flatten()

        t_eval = torch.arange(
            1, self.forecast_length + 1, 1, device=library_Theta.device
        ).float()
        library_Theta_flat = library_Theta.flatten()
        library_Theta_flat = library_Theta_flat.to(torch.cfloat)
        rollout = odeint(f, library_Theta_flat, t_eval, method="rk4")
        rollout = rollout.real
        rollout = rollout.reshape(
            self.forecast_length, library_Theta.shape[0], library_Theta.shape[1]
        )

        rollout = einops.rearrange(
            rollout,
            "n b h -> b n h",
            n=self.forecast_length,
            b=batch_size,
            h=hidden_size,
        )

        return rollout
