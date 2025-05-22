import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
from vanilla_transformer import Transformer
from david_ye_vanilla_transformer import TRANSFORMER
from mars_sindy_attention_transformer import TRANSFORMER_SINDY
from sindy_attention_transformer import SindyAttentionTransformer, SindyAttentionSindyLossTransformer
from sindy_loss_transformer import SINDyLossTransformer
from sindy_loss_rnns import SINDyLossGRU, SINDyLossLSTM
from rnns import GRU, LSTM
from decoders import MLP, UNET

from src import helpers


# Local files
pkg_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(pkg_path))

# Directories
top_dir = Path(__file__).parent.parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

def load_model_from_checkpoint(checkpoint_path, force_load=False, args=None):
    model = MixedModel(args)
    print("Checking if checkpoint exists")
    if (not args.skip_load_checkpoint or force_load) and checkpoint_path.exists():
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
        sensors = checkpoint['sensors']
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
        # Generate sensors
        # Handle SST differently (don't place sensors on land)
        if args.dataset == "sst":
            sensors = helpers.generate_sensor_positions(args.n_sensors*4, args.data_rows, args.data_cols)
            with open(data_dir / 'sst' / 'SST_zeros.pkl', 'rb') as f:
                zeros = pickle.load(f)
            sensors = [pos for pos in sensors if (zeros[pos[0], pos[1]] == False)]
            sensors = sensors[0:args.n_sensors]
        else:
            sensors = helpers.generate_sensor_positions(args.n_sensors, args.data_rows, args.data_cols)
    print()
    return model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses, sensors

class MixedModel(nn.Module):
    """
    Main function to generate mixes of models
    """
    def __init__(self, args): # Added SINDy params
        super().__init__()

        if args.encoder == "gru":
            self.encoder = GRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device
            )
        elif args.encoder == "sindy_loss_gru":
            self.encoder = SINDyLossGRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                poly_order=args.poly_order,
                include_sine=args.include_sine,
                sindy_regularization=args.sindy_weight,  # Use CLI argument
                sindy_threshold=args.sindy_threshold,    # Use CLI argument
                dt=args.dt,                            # Time step for Euler integration
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
        elif args.encoder == "sindy_loss_lstm":
            self.encoder = SINDyLossLSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                poly_order=args.poly_order,
                include_sine=args.include_sine,
                sindy_regularization=args.sindy_weight,  # Use CLI argument
                sindy_threshold=args.sindy_threshold,    # Use CLI argument
                dt=args.dt,                             # Time step for Euler integration
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
        elif args.encoder == "sindy_attention_sindy_loss_transformer":
            self.encoder = SindyAttentionSindyLossTransformer(
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
                sindy_regularization=args.sindy_weight,  # Use CLI argument
                sindy_threshold=args.sindy_threshold,    # Use CLI argument
                dt=args.dt,                              # Time step for Euler integration
                device=args.device
            )
        elif args.encoder == "sindy_loss_transformer":
            self.encoder = SINDyLossTransformer(
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
                device=args.device,
                sindy_regularization=args.sindy_weight,  # Use CLI argument
                sindy_threshold=args.sindy_threshold,    # Use CLI argument
                dt=args.dt                             # Time step for Euler integration
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
        else:
            raise NotImplementedError(f"Encoder {args.encoder} not implemented")
        
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
            raise NotImplementedError(f"Decoder {args.decoder} not implemented")

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
