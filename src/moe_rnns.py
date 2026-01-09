"""
Mixture of Experts RNN models with SINDy layer forecasting.
Combines GRU/LSTM encoders with multiple SINDy expert layers for interpretable dynamics.
"""

import torch
import torch.nn as nn
from sindy_layer import SindyLayer


class MOE_SINDy_Layer_Helpers_Mixin:
    """
    Mixin class providing helper methods for Mixture of Experts models with SINDy layers.

    Provides common functionality for printing, modifying, and analyzing SINDy
    coefficients across multiple expert networks.
    """

    def print_sindy_layer_coefficients(self):
        """
        Print the SINDy coefficients for all experts in a human-readable format.

        Displays the coefficient matrix as a polynomial expression for each
        hidden dimension of each expert.

        Returns:
            None
        """
        for j in range(self.n_experts):
            print(f"Expert {j}:")
            coefficients = self.experts[j].get_dense_sindy_coefficients()
            library = self.experts[j].pf.get_feature_names_out()
            for k in range(coefficients.shape[1]):
                print(f"Hidden layer {k}:")
                output_str = ""
                for l in range(coefficients.shape[0]):
                    output_str += (
                        f"{coefficients[l][k].item():.3f} \\cdot {library[l]} + "
                    )
                print(output_str[:-3])
            print()

    def set_forecast_length(self, forecast_length: int):
        """
        Set the forecast length for the model and all expert SINDy layers.

        Args:
            forecast_length (int): Number of timesteps to forecast

        Returns:
            None
        """
        self.forecast_length = forecast_length
        for expert in self.experts:
            expert.forecast_length = forecast_length

    def get_sindy_layer_coefficients_eigenvalues(self):
        """
        Get the eigenvalues of SINDy coefficient matrices for all experts.

        Returns:
            list: List of eigenvalue tensors, one per expert
        """
        with torch.no_grad():
            eigvs_l = []
            for i in range(self.n_experts):
                eigvs_l.append(self.experts[i].get_eigenvalues())
            return eigvs_l

    def get_sindy_layer_coefficients_sum(self):
        """
        Compute the sum of absolute SINDy coefficients across all experts.

        Used as a regularization term to encourage sparsity.

        Returns:
            float: Sum of square roots of absolute coefficient sums
        """
        with torch.no_grad():
            sindy_sum = 0.0
            for expert in self.experts:
                sindy_sum += torch.sqrt(
                    torch.abs(expert.get_raw_sindy_coefficients()).sum()
                )
            return sindy_sum

    def threshold_sindy_layer_coefficients(self, threshold, verbose=False):
        """
        Apply sparsity thresholding to SINDy coefficients for all experts.

        Sets coefficients with absolute value below the threshold to zero.

        Args:
            threshold (float): Threshold value; coefficients below this are zeroed
            verbose (bool): If True, print information about thresholding (default: False)

        Returns:
            None
        """
        with torch.no_grad():
            for i in range(self.n_experts):
                expert = self.experts[i]
                mask = torch.abs(expert.get_raw_sindy_coefficients()) > threshold
                expert.set_raw_sindy_coefficients(
                    expert.get_raw_sindy_coefficients() * mask
                )
                if verbose:
                    print(
                        f"MOE_SINDy_Layer_Helpers_Mixin: Applied threshold {threshold} to expert {i}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}"
                    )
        if verbose:
            print()


class MOEGRU(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
    """
    Mixture of Experts GRU with SINDy layer forecasting.

    Combines a GRU encoder with multiple SINDy expert layers for long-horizon
    forecasting. Expert outputs are combined via learned weighted averaging.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the MOE-GRU model.

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): Hidden state dimension for GRU and experts
            n_experts (int): Number of SINDy expert layers
            forecast_length (int): Number of timesteps to forecast
            strict_symmetry (bool): If True, enforce symmetric SINDy coefficients
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability for expert weighting
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.forecast_length = forecast_length
        self.strict_symmetry = strict_symmetry
        self.num_layers = num_layers
        self.gru = None  # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.initialize()

    def initialize(self):
        """
        Initialize the GRU, expert combination weights, and SINDy expert layers.

        Returns:
            None
        """
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # self.mixture_mlp = nn.Linear(self.n_experts)
        self.softmax = nn.Softmax(dim=-1)  # TODO: check dim
        self.linear_combination = nn.Parameter(
            torch.ones(self.n_experts) / self.n_experts
        )
        self.experts = nn.ModuleList(
            [
                SindyLayer(
                    d_model=self.hidden_size,
                    forecast_length=self.forecast_length,
                    device=self.device,
                    strict_symmetry=self.strict_symmetry,
                )
                for _ in range(self.n_experts)
            ]
        )

    def forward(self, x):
        """
        Forward pass through the MOE-GRU model.

        Processes input through the GRU, then passes the final hidden state
        through all SINDy experts and combines their outputs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            dict: Dictionary containing:
                - "sequence_output" (torch.Tensor): GRU output sequence of shape (batch_size, sequence_length, hidden_size)
                - "final_hidden_state" (torch.Tensor): Final GRU hidden state of shape (num_layers, batch_size, hidden_size)
                - "output" (torch.Tensor): Combined expert forecasts of shape
                    (batch_size, forecast_length, 1, hidden_size)
        """
        # Normal GRU forward
        out, h_out = self.gru(x)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": combined,
        }


class MOELSTM(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
    """
    Mixture of Experts LSTM with SINDy layer forecasting.

    Combines an LSTM encoder with multiple SINDy expert layers for long-horizon
    forecasting. Expert outputs are combined via learned weighted averaging.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        forecast_length: int,
        strict_symmetry: bool,
        num_layers: int,
        dropout: float,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the MOE-LSTM model.

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): Hidden state dimension for LSTM and experts
            n_experts (int): Number of SINDy expert layers
            forecast_length (int): Number of timesteps to forecast
            strict_symmetry (bool): If True, enforce symmetric SINDy coefficients
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability for expert weighting
            device (str): Device to place the model on (default: "cpu")
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.forecast_length = forecast_length
        self.strict_symmetry = strict_symmetry
        self.num_layers = num_layers
        self.lstm = None  # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.initialize()

    def initialize(self):
        """
        Initialize the LSTM, expert combination weights, and SINDy expert layers.

        Returns:
            None
        """
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # self.mixture_mlp = nn.Linear(self.n_experts)
        self.softmax = nn.Softmax(dim=-1)  # TODO: check dim
        self.linear_combination = nn.Parameter(
            torch.ones(self.n_experts) / self.n_experts
        )
        self.experts = nn.ModuleList(
            [
                SindyLayer(
                    d_model=self.hidden_size,
                    forecast_length=self.forecast_length,
                    device=self.device,
                    strict_symmetry=self.strict_symmetry,
                )
                for _ in range(self.n_experts)
            ]
        )

    def forward(self, x):
        """
        Forward pass through the MOE-LSTM model.

        Processes input through the LSTM, then passes the final hidden state
        through all SINDy experts and combines their outputs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            dict: Dictionary containing:
                - "sequence_output" (torch.Tensor): GRU output sequence of shape (batch_size, sequence_length, hidden_size)
                - "final_hidden_state" (torch.Tensor): Final GRU hidden state of shape (num_layers, batch_size, hidden_size)
                - "output" (torch.Tensor): Combined expert forecasts of shape
                    (batch_size, forecast_length, 1, hidden_size)
        """
        # Normal LSTM forward
        out, (h_out, c_out) = self.lstm(x)

        # SINDy forward all experts
        sindy_outputs = [expert(h_out[-1]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)
        sindy_outputs = sindy_outputs.unsqueeze(3)  # Adds sequence length dimension

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum("ebfsd,e->bfsd", sindy_outputs, weights)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out,
            "output": combined,
        }
