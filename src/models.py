import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import torch.nn as nn

from helpers import calculate_library_dim, sindy_library_torch

def load_model_from_checkpoint(args):
    model = MixedModel(args)
    if args.checkpoint_path.exists():
        checkpoint = torch.load(args.checkpoint_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val = checkpoint['best_val']
        if args.verbose:
            print(f"Loading model from {args.checkpoint_path}")
            print(f"> start_epoch: {start_epoch}")
            print(f"> best_val: {best_val:0.4e}")
    else:
        if args.verbose:
            print(f"Using newly initialized model")
        checkpoint=None
        start_epoch = 0
        best_val = float('inf')
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer, start_epoch, best_val

class PositionalEncoding(nn.Module):
    """
    source: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """
    def __init__(self, d_model: int, sequence_length: int = 5400, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(sequence_length, 1, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """
    Creates a simple linear MLP AutoEncoder.

    `in_dim`: input and output dimension   
    `bottleneck_dim`: dimension at bottleneck  
    `width`: width of model   
    `device`: which device to use   
    """
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, dropout: float, device: str = 'cpu'):
        super(MLP, self).__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Model layer sizes
        sizes = list()
        sizes.extend(np.logspace(np.log2(in_dim), np.log2(out_dim), base=2, num=n_layers+1, dtype=int).tolist())
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

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
        out = self.dropout(out)
        out = torch.mean(out, dim=1)
        return out

class UNET(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, dropout: float, device: str = 'cpu'):
        super().__init__()
        # Class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
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
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        out = torch.mean(out, dim=1)
        return out

class TRANSFORMER_SINDY(torch.nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.05,
                 poly_order: int = 1, include_sine: bool = False,
                 num_sindy_layers: int = 2, dim_feedforward: int = 128,
                 window_length: int = 500,
                 hidden_size: int = 64,
                 activation=nn.GELU(),
                 device:str='cpu'): # Added SINDy params
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.num_sindy_layers = num_sindy_layers
        self.window_length = window_length
        self.device = device

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

        self.initialize(self.d_model, self.window_length)

    def initialize(self, d_model:int, window_length:int, **kwargs):
        self.pos_encoder = PositionalEncoding(
            d_model=self.hidden_size,
            sequence_length=window_length + 10, # Provide some buffer
            dropout=self.dropout
        )

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
        x = self.pos_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # apply SINDy encoder/attention layers
        x = self.sindy_encoder(x) # Shape: (batch_size, seq_len, d_model)

        # dropout layer
        x = self.dropout_layer(x)

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

class TRANSFORMER(torch.nn.Module):
    """
    Standard Transformer Encoder model using nn.TransformerEncoderLayer.
    Uses GRU for input embedding as per the original user code structure.
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, # nhead=16 was high for d_model=128, using 8
                 num_encoder_layers: int = 1, dim_feedforward: int = 256, # Often 4*d_model
                 dropout: float = 0.2, activation = nn.GELU(),
                 window_length: int = 10, hidden_size: int = 10, device:str='cpu'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.window_length = window_length
        self.hidden_size = hidden_size
        self.device = device

        # --- Standard Transformer Components ---
        # Define a single encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True, # Important: assumes input shape (batch, seq, feature)
            norm_first=False   # Standard: Apply norm after Add; True applies before
        )
        # Stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_size) if num_encoder_layers > 1 else None # Optional final norm
        )
        # --- End Standard Transformer Components ---

        self.pos_encoder = None
        self.input_embedding = None

        self.output_size = d_model # Output dim is the model's hidden dim

        self.initialize(self.hidden_size, self.window_length)

    def initialize(self, hidden_size: int, lags: int, **kwargs):
        """Initialize components that depend on input dimensions and sequence length."""
        sequence_length = lags # Use 'lags' as sequence_length

        # Ensure d_model is divisible by nhead
        if self.hidden_size % self.nhead != 0:
             raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by nhead ({self.nhead})")

        # Initialize Positional Encoding with correct sequence length awareness
        # Note: max_sequence_length in PositionalEncoding should be >= lags
        self.pos_encoder = PositionalEncoding(
            d_model=self.hidden_size,
            sequence_length=sequence_length + 10, # Provide some buffer
            dropout=self.dropout
        )

        # Initialize Input Embedding (using GRU as per original structure)
        self.input_embedding = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_size, # GRU output matches d_model
            num_layers=2,                 # Example: 2 GRU layers for embedding
            batch_first=True,
            dropout=self.dropout if self.num_encoder_layers > 1 else 0.0 # Dropout between GRU layers
        )

    def _generate_square_subsequent_mask(self, sequence_length: int, device) -> torch.Tensor:
        """Generates an upper-triangular matrix mask for causal attention."""
        # Creates a mask where future positions are masked (-inf or True)
        # torch.triu: upper triangle of a matrix. diagonal=1 means diagonal is 0 (not masked)
        mask = torch.triu(torch.full((sequence_length, sequence_length), float('-inf'), device=device), diagonal=1)
        # For nn.TransformerEncoderLayer's 'src_mask', float('-inf') is standard.
        # If 'attn_mask' in nn.MultiheadAttention is used directly, it often expects boolean (True means ignore)
        return mask # Shape: [sequence_length, sequence_length]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Dictionary containing sequence output and final hidden state
        """
        if self.input_embedding is None or self.pos_encoder is None:
            raise RuntimeError("Model must be initialized before calling forward.")

        # 1. Input Embedding
        # GRU returns output sequence and hidden state(s)
        # We only need the output sequence here
        x_embedded, _ = self.input_embedding(x) # Shape: (batch_size, seq_len, d_model)

        # 2. Add Positional Encoding
        x_pos_encoded = self.pos_encoder(x_embedded) # Shape: (batch_size, seq_len, d_model)

        # 3. Generate Causal Mask
        # Mask prevents attention to future positions. Shape: [seq_len, seq_len]
        # Needs to be on the same device as the input tensor 'x'.
        causal_mask = self._generate_square_subsequent_mask(x_pos_encoded.size(1), x_pos_encoded.device)

        # 4. Apply Transformer Encoder Layers
        # The mask ensures causality.
        transformer_output = self.transformer_encoder(
            src=x_pos_encoded,
            mask=causal_mask,
            src_key_padding_mask=None # Optional: for masking padded elements
        ) # Shape: (batch_size, seq_len, d_model)

        # 5. Apply dropout
        transformer_output = self.dropout_layer(transformer_output)

        # 5. Prepare Output
        return {
            "sequence_output": transformer_output, # [batch_size, sequence_length, d_model]
            "final_hidden_state": transformer_output[:, -1, :] # Last timestep [batch_size, d_model]
        }

class LSTM(torch.nn.Module):
    def __init__(self, input_size:int = 3, hidden_size:int = 64, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        device = next(self.parameters()).device
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        out = self.dropout(out)
        h_out = self.dropout(h_out)

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size)
        }

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
                num_sindy_layers=args.encoder_depth,
                dim_feedforward=args.dim_feedforward,
                window_length=args.window_length,
                hidden_size=args.hidden_size,
                activation=nn.GELU(),
                device=args.device
            )
        elif args.encoder == "sindy_loss_transformer":
            raise NotImplementedError
        elif args.encoder == "transformer":
            self.encoder = TRANSFORMER(
                d_model=args.d_model,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                window_length=args.window_length,
                num_encoder_layers=args.encoder_depth,
                device=args.device
            )
        elif args.encoder == "lstm":
            self.encoder = LSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device
            )
        else:
            raise NotImplementedError
        
        if args.decoder == "unet":
            self.decoder = UNET(
                in_dim = args.hidden_size,
                out_dim = args.output_size,
                n_layers = args.decoder_depth,
                dropout=args.dropout,
                device=args.device
            )
        elif args.decoder == "mlp":
            self.decoder = MLP(
                in_dim = args.hidden_size,
                out_dim = args.output_size,
                n_layers = args.decoder_depth,
                dropout=args.dropout,
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
