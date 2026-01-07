import torch
import einops
import torch.nn as nn
from torchdiffeq import odeint
from pytorch_polynomial_features import PolynomialFeatures
from sindy_layer import SindyLayer

class MOEGRU(nn.Module):
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

    def set_forecast_length(self, forecast_length: int):
        self.forecast_length = forecast_length
        for expert in self.experts:
            expert.forecast_length = forecast_length

    def initialize(self):
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        #self.mixture_mlp = nn.Linear(self.n_experts)
        self.softmax = nn.Softmax(dim=-1)  # TODO: check dim
        self.linear_combination = nn.Parameter(torch.ones(self.n_experts)/self.n_experts)
        self.experts = nn.ModuleList([SindyLayer(
                d_model=self.hidden_size,
                forecast_length=self.forecast_length,
                device=self.device,
                strict_symmetry=self.strict_symmetry,
            ) for _ in range(self.n_experts)])

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        # Normal GRU forward
        out, h_out = self.gru(x)

        # SINDy forward all experts
        # TODO: What should input to expert be? What should output of model be?
        sindy_outputs = [expert(h_out[-1:]) for expert in self.experts]
        sindy_outputs = torch.stack(sindy_outputs)

        # Combine experts: weighted sum across expert dimension
        # NOTE: Dropout drops random experts
        weights = self.dropout(self.linear_combination)
        weights = self.softmax(weights)
        combined = torch.einsum('ebfsd,e->bfsd', sindy_outputs, weights)

        return {
            "sequence_output": out,  # [batch_size, forecast_length, sequence_length, d_model]
            "final_hidden_state": h_out,  # [batch_size, 1, encoder_depth, d_model]
            "output": combined,
        }
