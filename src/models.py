"""
Model architecture and checkpoint loading utilities.
Provides the MixedModel wrapper for encoder-decoder combinations and checkpoint management.
"""

import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
from vanilla_transformer import Transformer
from sindy_attention_transformer import SindyAttentionTransformer
from sindy_attention_sindy_loss_transformer import SindyAttentionSindyLossTransformer
from sindy_loss_transformer import SINDyLossTransformer
from sindy_loss_rnns import SINDyLossGRU, SINDyLossLSTM, SINDyLossMLP
from rnns import GRU, LSTM, MLPEncoder
from decoders import MLPDecoder, CNN
from moe_rnns import MOEGRU, MOELSTM, MOEMLP
from sindy_loss_moe_rnns import SINDyLossMOEGRU, SINDyLossMOEMLP, SINDyLossMOELSTM

from src import helpers

# Local files
pkg_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(pkg_path))

# Directories
top_dir = Path(__file__).parent.parent
data_dir = top_dir / "datasets"
plasma_dir = data_dir / "plasma"
fig_dir = top_dir / "figures"


def load_model_from_checkpoint(checkpoint_path, force_load=False, args=None):
    """
    Load a model from a checkpoint file or initialize a new model.

    If a checkpoint exists and loading is not skipped, restores the model state,
    optimizer state, training history, and sensor positions. Otherwise, creates
    a fresh model with newly generated sensor positions.

    Args:
        checkpoint_path (Path): Path to the checkpoint file
        force_load (bool): If True, load checkpoint even if skip_load_checkpoint is set (default: False)
        args (argparse.Namespace): Configuration arguments containing model hyperparameters

    Returns:
        tuple: A tuple containing:
            - model (MixedModel): The loaded or newly initialized model
            - optimizer (torch.optim.Adam): Optimizer with appropriate parameter groups
            - start_epoch (int): Epoch to resume training from
            - best_val (float): Best validation loss achieved
            - best_epoch (int): Epoch when best validation was achieved
            - train_losses (list): History of training losses
            - val_losses (list): History of validation losses
            - model_eigvs (list): History of model eigenvalues (for SINDy models)
            - sensors (list): List of (row, col) sensor position tuples
    """
    model = MixedModel(args)
    print("Checking if checkpoint exists")
    if (not args.skip_load_checkpoint or force_load) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        coefficient_params = [
            p
            for name, p in model.named_parameters()
            if "triangle_coefficients" in name or "sindy_coefficients" in name
        ]
        other_params = [
            p
            for name, p in model.named_parameters()
            if not ("triangle_coefficients" in name or "sindy_coefficients" in name)
        ]

        if args.coord_descent:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": coefficient_params,
                        "lr": args.coord_descent_sindy_layer_lr,
                    },
                    {"params": other_params, "lr": args.coord_descent_model_lr},
                ]
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint["best_val"]
        best_epoch = checkpoint["best_epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        model_eigvs = checkpoint["model_eigvs"]
        sensors = checkpoint["sensors"]
        if args.verbose:
            print(f"Loading model from {checkpoint_path}")
            print(f"> start_epoch: {start_epoch}")
            print(f"> best_val: {best_val:0.4e}")
    else:
        if args.verbose:
            print(f"Using newly initialized model")
        checkpoint = None
        start_epoch = 0
        best_val = float("inf")
        model.to(args.device)

        coefficient_params = [
            p
            for name, p in model.named_parameters()
            if "self_attn.coefficients" in name
        ]
        other_params = [
            p
            for name, p in model.named_parameters()
            if "self_attn.coefficients" not in name
        ]

        if args.coord_descent:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": coefficient_params,
                        "lr": args.coord_descent_sindy_layer_lr,
                    },
                    {"params": other_params, "lr": args.coord_descent_model_lr},
                ]
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        val_losses = []
        model_eigvs = []
        best_epoch = 0
        # Generate sensors
        # Handle SST differently (don't place sensors on land)
        if args.dataset == "sst":
            sensors = helpers.generate_sensor_positions(
                args.n_sensors * 4, args.data_rows_in, args.data_cols_in
            )
            with open(data_dir / "sst" / "SST_zeros.pkl", "rb") as f:
                zeros = pickle.load(f)
            sensors = [pos for pos in sensors if (zeros[pos[0], pos[1]] == False)]
            sensors = sensors[0 : args.n_sensors]
        else:
            sensors = helpers.generate_sensor_positions(
                args.n_sensors, args.data_rows_in, args.data_cols_in
            )
    if args.verbose:
        print()
    return (
        model,
        optimizer,
        start_epoch,
        best_val,
        best_epoch,
        train_losses,
        val_losses,
        model_eigvs,
        sensors,
    )


class MixedModel(nn.Module):
    """
    A flexible encoder-decoder model supporting multiple encoder and decoder types.

    Combines various encoder architectures (RNNs, Transformers, SINDy variants)
    with decoder architectures (MLP, CNN) based on configuration arguments.
    """

    def __init__(self, args):
        """
        Initialize the MixedModel with specified encoder and decoder.

        Args:
            args (argparse.Namespace): Configuration arguments containing:
                - encoder (str): Encoder type ("gru", "lstm", "mlp", "moe_gru",
                    "moe_lstm", "moe_mlp", "sindy_loss_gru", "sindy_loss_lstm", "sindy_loss_mlp",
                    "sindy_loss_moe_gru", "sindy_loss_moe_mlp", "sindy_loss_moe_lstm",
                    "vanilla_transformer", "sindy_attention_transformer",
                    "sindy_attention_sindy_loss_transformer", "sindy_loss_transformer")
                - decoder (str): Decoder type ("mlp" or "cnn")
                - d_model (int): Input dimension
                - hidden_size (int): Hidden layer size
                - encoder_depth (int): Number of encoder layers
                - decoder_depth (int): Number of decoder layers
                - dropout (float): Dropout probability
                - device (str): Device to place the model on
                - And other encoder-specific parameters (n_heads, forecast_length, etc.)
        """
        super().__init__()

        if args.encoder == "mlp":
            self.encoder = MLPEncoder(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "sindy_loss_mlp":
            self.encoder = SINDyLossMLP(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                device=args.device,
            )
        elif args.encoder == "sindy_loss_moe_mlp":
            self.encoder = SINDyLossMOEMLP(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                num_layers=args.encoder_depth,
                forecast_length=args.forecast_length,
                strict_symmetry=args.strict_symmetry,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "gru":
            self.encoder = GRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "sindy_loss_gru":
            self.encoder = SINDyLossGRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                device=args.device,
            )
        elif args.encoder == "moe_gru":
            self.encoder = MOEGRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                forecast_length=args.forecast_length,
                num_layers=args.encoder_depth,
                strict_symmetry=args.strict_symmetry,
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "sindy_loss_moe_gru":
            self.encoder = SINDyLossMOEGRU(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                num_layers=args.encoder_depth,
                forecast_length=args.forecast_length,
                strict_symmetry=args.strict_symmetry,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "moe_lstm":
            self.encoder = MOELSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                forecast_length=args.forecast_length,
                num_layers=args.encoder_depth,
                strict_symmetry=args.strict_symmetry,
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "moe_mlp":
            self.encoder = MOEMLP(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                forecast_length=args.forecast_length,
                num_layers=args.encoder_depth,
                strict_symmetry=args.strict_symmetry,
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "lstm":
            self.encoder = LSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "sindy_loss_lstm":
            self.encoder = SINDyLossLSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_depth,
                dropout=args.dropout,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                device=args.device,
            )
        elif args.encoder == "sindy_loss_moe_lstm":
            self.encoder = SINDyLossMOELSTM(
                input_size=args.d_model,
                hidden_size=args.hidden_size,
                n_experts=args.n_experts,
                num_layers=args.encoder_depth,
                forecast_length=args.forecast_length,
                strict_symmetry=args.strict_symmetry,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,  # Time step for Euler integration
                std_init_min=args.std_init_min,
                std_init_max=args.std_init_max,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.encoder == "vanilla_transformer":
            self.encoder = Transformer(
                d_model=args.d_model,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                input_length=args.input_length,
                num_encoder_layers=args.encoder_depth,
                norm_first=False,
                layer_norm_eps=1e-5,
                bias=True,
                device=args.device,
            )
        elif args.encoder == "sindy_attention_transformer":
            self.encoder = SindyAttentionTransformer(
                d_model=args.d_model,
                n_heads=args.n_heads,
                forecast_length=args.forecast_length,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                input_length=args.input_length,
                num_encoder_layers=args.encoder_depth,
                layer_norm_eps=1e-5,
                norm_first=False,
                strict_symmetry=args.strict_symmetry,
                bias=True,
                device=args.device,
            )
        elif args.encoder == "sindy_attention_sindy_loss_transformer":
            self.encoder = SindyAttentionSindyLossTransformer(
                d_model=args.d_model,
                n_heads=args.n_heads,
                forecast_length=args.forecast_length,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                input_length=args.input_length,
                num_encoder_layers=args.encoder_depth,
                layer_norm_eps=1e-5,
                norm_first=False,
                bias=True,
                strict_symmetry=args.strict_symmetry,
                poly_order=args.poly_order,
                sindy_loss_threshold=args.sindy_loss_threshold,
                dt=args.dt,
                device=args.device,
            )
        elif args.encoder == "sindy_loss_transformer":
            self.encoder = SINDyLossTransformer(
                d_model=args.d_model,
                n_heads=args.n_heads,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
                activation=nn.GELU(),
                hidden_size=args.hidden_size,
                input_length=args.input_length,
                num_encoder_layers=args.encoder_depth,
                layer_norm_eps=1e-5,
                norm_first=False,
                bias=True,
                poly_order=args.poly_order,
                device=args.device,
                sindy_loss_threshold=args.sindy_loss_threshold,  # Use CLI argument
                dt=args.dt,  # Time step for Euler integration
            )
        else:
            raise NotImplementedError(f"Encoder {args.encoder} not implemented")

        if args.decoder == "cnn":
            self.decoder = CNN(
                in_dim=args.hidden_size,
                out_dim=args.output_size,
                n_layers=args.decoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.decoder == "mlp":
            self.decoder = MLPDecoder(
                in_dim=args.hidden_size,
                out_dim=args.output_size,
                n_layers=args.decoder_depth,
                dropout=args.dropout,
                device=args.device,
            )
        else:
            raise NotImplementedError(f"Decoder {args.decoder} not implemented")

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)

    def forward(self, src: torch.Tensor) -> dict:
        """
        Forward pass through the encoder-decoder model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_sensors * d_input)

        Returns:
            dict: Dictionary containing:
                - "output" (torch.Tensor): Decoded output tensor
                - "sindy_loss" (torch.Tensor or None): SINDy loss if applicable
        """
        src_encoded = self.encoder(src)
        src_decoded = self.decoder(src_encoded)

        return src_decoded
