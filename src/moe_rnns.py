import torch
import torch.nn as nn
from sindy_layer import SindyLayer


class MOE_SINDy_Layer_Helpers_Mixin:
    """Mixin class providing helper methods for MOE models with SINDy layers."""

    def print_sindy_layer_coefficients(self):
        # coefficients: n_heads x ((library terms + 1 (for linear) terms) x library_terms equations)
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
        self.forecast_length = forecast_length
        for expert in self.experts:
            expert.forecast_length = forecast_length

    def get_sindy_layer_coefficients_eigenvalues(self):
        with torch.no_grad():
            eigvs_l = []
            for i in range(self.n_experts):
                eigvs_l.append(self.experts[i].get_eigenvalues())
            return eigvs_l

    def get_sindy_layer_coefficients_sum(self):
        with torch.no_grad():
            sindy_sum = 0.0
            for expert in self.experts:
                sindy_sum += torch.sqrt(
                    torch.abs(expert.get_raw_sindy_coefficients()).sum()
                )
            return sindy_sum

    def threshold_sindy_layer_coefficients(self, threshold, verbose=False):
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
        Forward pass through the GRU model.
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
            "sequence_output": out,  # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out,  # [batch_size, 1, encoder_depth, d_model]
            "output": combined,
        }


class MOELSTM(nn.Module, MOE_SINDy_Layer_Helpers_Mixin):
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
        Forward pass through the GRU model.
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
            "sequence_output": out,  # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out,  # [batch_size, 1, encoder_depth, d_model]
            "output": combined,
        }
