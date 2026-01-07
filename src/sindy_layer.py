import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
from pytorch_polynomial_features import PolynomialFeatures

class SindyLayer(nn.Module):
    """
    Sindy Layer is a module that fits an ODE using SINDy to the data, and then is capable of arbitrary-length forecasting using the fitted ODE.

    Can either enforce strict symmetry, or allow for general symmetry.
    """
    def __init__(
        self,
        d_model: int,
        forecast_length: int,
        device: str = "cpu",
        strict_symmetry: bool = True,
        **kwargs,
    ):
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
        # TODO: Initialization? Should be larger I think, or different per-expert?
        if self.strict_symmetry:
            # Symmetric parameters, builds a 1D list of parameters (lower triangle) that get converted to a dense symmetric matrix
            self.tril_indices = torch.tril_indices(self.library_dim, self.library_dim)
            num_params = (self.library_dim * (self.library_dim + 1)) // 2
            self.triangle_coefficients = nn.Parameter(torch.Tensor(num_params))
            nn.init.normal_(self.triangle_coefficients, mean=0.0, std=0.1) 
        else:
            # General dense 1D matrix, initialized as a symmetric matrix
            num_params = self.library_dim * self.library_dim
            triangle_coefficients = torch.randn(num_params)*0.1 + 0.0
            sindy_coefficients = self.dense_matrix_from_symmetric_params(triangle_coefficients)
            self.sindy_coefficients = nn.Parameter(sindy_coefficients)

    def get_dense_sindy_coefficients(self) -> torch.Tensor:
        """
        Convert symmetric parameters (1D list) to a dense matrix.
        Only used when `strict_symmetry` is True.
        """
        if self.strict_symmetry:
            sindy_coefficients = self.dense_matrix_from_symmetric_params(self.triangle_coefficients)
            return sindy_coefficients
        else:
            return self.sindy_coefficients

    def dense_matrix_from_symmetric_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert symmetric parameters (1D list) to a dense matrix.
        Only used when `strict_symmetry` is True.
        """
        sindy_coefficients = torch.zeros(self.library_dim, self.library_dim, device=params.device)
        self.tril_indices = self.tril_indices.to(sindy_coefficients.device)
        sindy_coefficients[self.tril_indices[0], self.tril_indices[1]] = params
        sindy_coefficients = sindy_coefficients + sindy_coefficients.t() - torch.diag(sindy_coefficients.diag())
        return sindy_coefficients

    def get_eigenvalues(self) -> torch.Tensor:
        """
        Get the eigenvalues of the SINDy coefficients.
        """
        sindy_coefficients = self.get_dense_sindy_coefficients()
        eigenvalues = torch.linalg.eigvals(sindy_coefficients.to(torch.cfloat) * torch.tensor(1j))
        return eigenvalues

    def forward(self, x):
        """ """
        seq_len, batch_size, hidden_size = x.shape
        sindy_coefficients = self.get_dense_sindy_coefficients()
        x_flat = einops.rearrange(x, "b s h -> (b s) h")
        library_Theta = self.pf.fit_transform(x_flat)

        def f(t, y):
            y = y.reshape(library_Theta.shape[0], library_Theta.shape[1])
            y = y.T
            terms = sindy_coefficients.to(y.device) 
            terms = terms.to(torch.cfloat) * torch.tensor(1j)
            dy = terms @ y
            dy = dy.T
            return dy.flatten()
        
        t_eval = torch.arange(1, self.forecast_length+1, 1, device=library_Theta.device).float()
        library_Theta_flat = library_Theta.flatten()
        library_Theta_flat = library_Theta_flat.to(torch.cfloat)
        rollout = odeint(f, library_Theta_flat, t_eval, method='rk4')
        rollout = rollout.real
        rollout = rollout.reshape(self.forecast_length, library_Theta.shape[0], library_Theta.shape[1])

        # Reshape update back to (forecast, batch_size, seq_len, hidden_size)
        rollout = einops.rearrange(rollout, 'n (b s) h -> b n s h', n=self.forecast_length, b=batch_size, s=seq_len,  h=hidden_size)

        return rollout
