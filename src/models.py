import math
import copy
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

from vanilla_transformer import Transformer
from david_ye_vanilla_transformer import TRANSFORMER
from mars_sindy_attention_transformer import TRANSFORMER_SINDY
from sindy_attention_transformer import SindyAttentionTransformer

def load_model_from_checkpoint(checkpoint_path, args):
    model = MixedModel(args)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val = checkpoint['best_val']
        best_epoch = checkpoint['best_epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        if args.verbose:
            print(f"Loading model from {checkpoint_path}")
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
        train_losses = []
        val_losses = []
        best_epoch = 0
    print()
    return model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses

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
        x = x["final_hidden_state"]
        out = self.model(x)
        out = self.dropout(out)
        return out

class UNET(nn.Module):
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
        sizes[0] = self.in_dim
        sizes[-1] = self.out_dim

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
        x = x["final_hidden_state"]
        x = x.permute(0, 2, 1)
        out = self.model(x)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size:int = 3, hidden_size:int = 64, num_layers:int = 2, dropout:float = 0.1, device:str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size
        self.dropout = nn.Dropout(dropout)

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

class MixedModel(nn.Module):
    """
    Main function to generate mixes of models
    """
    def __init__(self, args): # Added SINDy params
        super().__init__()

        if args.encoder == "lstm":
            self.encoder = LSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device
            )
        elif args.encoder == "vanilla_transformer":
            self.encoder = Transformer(
                d_model=args.d_model,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                window_length=args.window_length,
                num_encoder_layers=args.encoder_depth,
                layer_norm_eps=1e-5,
                bias=True,
                device=args.device
            )
        elif args.encoder == "david_ye_transformer":
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
        elif args.encoder == "sindy_attention_transformer":
            self.encoder = SindyAttentionTransformer(
                d_model=args.d_model,
                nhead=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                window_length=args.window_length,
                num_encoder_layers=args.encoder_depth,
                layer_norm_eps=1e-5,
                bias=True,
                poly_order=args.poly_order,
                include_sine=args.include_sine,
                device=args.device
            )
        elif args.encoder == "mars_sindy_attention_transformer":
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
