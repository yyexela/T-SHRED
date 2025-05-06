import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import torch.nn as nn

from helpers import calculate_library_dim, sindy_library_torch

class MLP(nn.Module):
    """
    Creates a simple linear MLP AutoEncoder.

    `in_dim`: input and output dimension   
    `bottleneck_dim`: dimension at bottleneck  
    `width`: width of model   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, device: str = 'cpu'):
        super(MLP, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[idx], sizes[idx+1]))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        x = x["sequence_output"]
        out = self.model(x)
        out = torch.mean(out, dim=1)
        return out

class UNET(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, device: str = 'cpu'):
        super().__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())

        # Define model layers
        self.layers = []
        for idx in range(len(sizes)-1):
            self.layers.append(nn.Conv1d(sizes[idx], sizes[idx+1], kernel_size=3, padding=1))
            if idx != (len(sizes)-2):
                self.layers.append(nn.ReLU())

        model = nn.Sequential(*self.layers)
        model = model.to(device)
        self.model = model

    def forward(self, x):
        x = x["sequence_output"]
        x = x.permute(0, 2, 1)
        out = self.model(x)
        out = out.permute(0, 2, 1)
        out = torch.mean(out, dim=1)
        return out

class TRANSFORMER_SINDY(torch.nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.05,
                 poly_order: int = 1, include_sine: bool = False,
                 num_sindy_layers: int = 2, dim_feedforward: int = 128,
                 window_length: int = 500,
                 hidden_size: int = 64,
                 activation=nn.GELU()): # Added SINDy params
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.num_sindy_layers = num_sindy_layers
        self.window_length = window_length

        # Store config for layers
        self.layer_config = {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "poly_order": poly_order,
            "include_sine": include_sine
        }

        self.output_size = d_model

        self.initialize(self.d_model, self.window_length)

    def initialize(self, d_model:int, window_length:int, **kwargs):
        #self.pos_encoder = PositionalEncoding(self.d_model,
                                              #window_length, # Use actual sequence length
                                              #self.dropout)

        # Input GRU embedding
        self.input_embedding = nn.GRU(input_size=self.d_model,
                                      hidden_size=self.hidden_size,
                                      num_layers=2, # Or adjust as needed
                                      batch_first=True,
                                      dropout=self.dropout if self.num_sindy_layers > 1 else 0.0) # Add dropout if multiple layers

        # Create the stack of SINDy layers (units) attention
        layers = [SINDyLayer(**self.layer_config) for _ in range(self.num_sindy_layers)]
        self.sindy_encoder = nn.Sequential(*layers) # Use nn.Sequential for simplicity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Dictionary containing sequence output and final hidden state
        """
        # apply input embedding
        # GRU output: (output_seq, final_hidden_state)
        x, _ = self.input_embedding(x) # Shape: (batch_size, seq_len, d_model)

        # perform positional encoding
        #x = self.pos_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # apply SINDy encoder/attention layers
        x = self.sindy_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # Return dictionary format as before
        return {
            "sequence_output": x, # [batch_size, sequence_length, d_model]
            "final_hidden_state": x[:,-1,:] # last timestep hidden state [batch_size, d_model]
        }

class SINDyLayer(torch.nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: nn.Module,
                 poly_order: int, include_sine: bool, sindy_threshold: float = 0.01, hidden_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = calculate_library_dim(hidden_size, poly_order, include_sine)
        self.sindy_threshold_val = sindy_threshold # For thresholding (sparsify)

        # SINDy coefficients (nn.parameter which is learnable)
        self.coefficients = nn.Parameter(torch.Tensor(self.library_dim, self.hidden_size))
        nn.init.xavier_uniform_(self.coefficients) # Initialize coefficients

        # Coefficient mask (not learnable, used for thresholding)
        self.register_buffer('coefficient_mask', torch.ones(self.library_dim, self.hidden_size), persistent=False)

        # Standard Transformer components
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.activation = activation


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, hidden_size)
        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_len, hidden_size = src.shape
        src_residual = src

        ############################# SINDy unit #############################
        # Reshape src for sindy_library: (batch_size * seq_len, hidden_size)
        src_flat = src.reshape(-1, hidden_size)

        # Calculate SINDy library features
        library_Theta = sindy_library_torch(src_flat, self.hidden_size, self.poly_order, self.include_sine)

        # Calculate SINDy update (use masked coefficients)
        # effective_coefficients = self.coefficients * self.coefficient_mask.to(self.coefficients.device) # Ensure mask is on correct device
        ############################## Simplified SINDy update (without mask) #############################
        sindy_update = library_Theta @ self.coefficients


        # Reshape update back to (batch_size, seq_len, hidden_size)
        sindy_update = sindy_update.reshape(batch_size, seq_len, hidden_size)
        ############################## End SINDy unit #############################

        # Apply dropout and skip connection + normalization
        src = self.norm1(src_residual + self.dropout1(sindy_update))

        ############################## MLP Block##############################
        src_residual = src
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        ######################################################################

        # dropout and skip connection + norm
        src = self.norm2(src_residual + self.dropout3(ff_output))

        return src

    # may cause numerical instability, temp optional
    def thresholding(self, threshold=None):
        if threshold is None:
            threshold = self.sindy_threshold_val
        with torch.no_grad():
            mask = torch.abs(self.coefficients.data) > threshold
            self.coefficients.data *= mask
            self.coefficient_mask.copy_(mask.float()) # Update buffer if needed for inspection
            print(f"SINDyLayer: Applied threshold {threshold}. Non-zero coeffs: {mask.sum().item()}/{mask.numel()}")

class MixedModel(torch.nn.Module):
    """
    Main function to generate mixes of models
    """
    def __init__(self, args): # Added SINDy params
        super().__init__()

        if args.encoder == "sindy_attention_transformer":
            self.encoder = TRANSFORMER_SINDY(
                d_model=args.d_model,
                dropout=args.dropout,
                poly_order=args.poly_order,
                include_sine=args.include_sine,
                num_sindy_layers=args.num_sindy_layers,
                dim_feedforward=args.dim_feedforward,
                window_length=args.window_length,
                hidden_size=args.hidden_size,
                activation=nn.GELU()
            )
        elif args.encoder == "sindy_loss_transformer":
            raise NotImplementedError
        elif args.encoder == "sindy_loss_transformer":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        if args.decoder == "unet":
            self.decoder = UNET(
                in_dim = args.hidden_size,
                out_dim = args.output_size,
                n_layers = args.decoder_depth,
                device=args.device
            )
        elif args.decoder == "mlp":
            self.decoder = MLP(
                in_dim = args.hidden_size,
                out_dim = args.output_size,
                n_layers = args.decoder_depth,
                device=args.device
            )
        else:
            raise NotImplementedError

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (batch_size, sequence_length, n_sensors, d_model)
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_sensors, d_model)
        """

        src_encoded = self.encoder(src)
        src_decoded = self.decoder(src_encoded)

        return src_decoded
