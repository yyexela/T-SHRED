"""
Helper utilities for training, evaluation, and data processing.
Includes argument parsing, model training loops, SVD/POD operations, and parallel execution.
"""

import os
import copy
import yaml
import time
import torch
import einops
import random
import pickle
import argparse
import threading
import subprocess
import numpy as np
import paramiko
from torch import nn
from queue import Queue, Empty
from pathlib import Path
from src.plots import plot_losses, plot_field_comparison


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # To allow CLAs
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=6, help="Dataset batch size")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for model parameters, overwrites other CLAs",
    )
    parser.add_argument(
        "--coord_descent",
        action="store_true",
        help="Use coordinate descent to train SINDy-Attention model",
    )
    parser.add_argument(
        "--coord_descent_model_n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train model except for SINDy-Attention coefficients during coordinate descent",
    )
    parser.add_argument(
        "--coord_descent_sindy_layer_n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train SINDy layer coefficients during coordinate descent",
    )
    parser.add_argument(
        "--coord_descent_model_lr",
        type=float,
        default=1e-3,
        help="Learning rate for model during coordinate descent",
    )
    parser.add_argument(
        "--coord_descent_sindy_layer_lr",
        type=float,
        default=1e-3,
        help="Learning rate for SINDy layer coefficients during coordinate descent",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to run (planetswe, sst, sst_demo, plasma)",
    )
    parser.add_argument(
        "--decoder", type=str, default="mlp", help="Which decoder to use (cnn, mlp)"
    )
    parser.add_argument(
        "--decoder_depth", type=int, default=1, help="Number of decoder layers"
    )
    parser.add_argument(
        "--delete_checkpoint",
        action="store_true",
        help="Delete checkpoint after training",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:2", help="Which device to run on"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Model droput proportion"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step for SINDy derivatives (Euler integration)",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=0,
        help="Train the model for at least this many epochs before saving best validation score",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="transformer",
        help="Which encoder to use (lstm, gru, sindy_loss_lstm, sindy_loss_gru, vanilla_transformer, sindy_attention_transformer, sindy_attention_sindy_loss_transformer)",
    )
    parser.add_argument(
        "--encoder_depth", type=int, default=1, help="Number of encoder layers"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--forecast_length",
        type=int,
        default=1,
        help="Number of timesteps to forecast (sindy_attention_transformer and sindy_attention_sindy_loss_transformer only)",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=12, help="Hidden size of encoder"
    )
    parser.add_argument(
        "--generate_test_plots", action="store_true", help="Generate test plots"
    )
    parser.add_argument(
        "--generate_training_plots", action="store_true", help="Generate training plots"
    )
    parser.add_argument(
        "--generate_loss_plots", action="store_true", help="Generate loss plots"
    )
    parser.add_argument(
        "--identifier", type=str, default=None, help="Identifier for logging"
    )
    parser.add_argument(
        "--input_length", type=int, default=10, help="Dataset window length"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for training"
    )
    parser.add_argument(
        "--n_experts", type=int, default=2, help="Number of experts for MOE-GRU"
    )
    parser.add_argument(
        "--n_heads", type=int, default=6, help="Number of transformer heads"
    )
    parser.add_argument("--n_sensors", type=int, default=50, help="Number of sensors")
    parser.add_argument(
        "--n_well_tracks",
        type=int,
        default=10,
        help="Maximum number of tracks to load from the well dataset",
    )
    parser.add_argument(
        "--poly_order",
        type=int,
        default=2,
        help="Order of polynomial library for SINDy transformer library in SINDy loss",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=10,
        help="After how many epochs to checkpoint model",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--sindy_layer_threshold",
        type=float,
        default=0.05,
        help="Threshold for SINDy coefficient sparsification (attention)",
    )
    parser.add_argument(
        "--sindy_layer_threshold_n_epochs",
        type=int,
        default=10,
        help="Every n epochs to threshold SINDy coefficients (attention)",
    )
    parser.add_argument(
        "--sindy_layer_weight",
        type=float,
        default=0.0,
        help="Weight for SINDy attention coefficient loss term",
    )
    parser.add_argument(
        "--sindy_loss_threshold",
        type=float,
        default=0.05,
        help="Threshold for SINDy coefficient sparsification loss",
    )
    parser.add_argument(
        "--sindy_loss_weight",
        type=float,
        default=100,
        help="Weight for SINDy loss term",
    )
    parser.add_argument(
        "--skip_load_checkpoint", action="store_true", help="Skip loading checkpoint"
    )
    parser.add_argument(
        "--strict_symmetry",
        action="store_true",
        help="Use strict symmetry for SINDy coefficients in SINDy layers",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose messages"
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        config_path = args.config
        args = argparse.Namespace(**model_config)
        args.config = config_path

    return args


def verify_args(args):
    """
    Validate that model parameters are within acceptable ranges.

    Checks all hyperparameters and configuration values for validity,
    raising ValueError if any parameter is invalid.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing model configuration

    Raises:
        ValueError: If any parameter is outside its valid range or incompatible with other settings
    """
    if args.batch_size <= 0:
        raise ValueError(f"batch_size {args.batch_size} must be greater than 0")
    if args.coord_descent and not ("sindy_attention" in args.encoder):
        raise ValueError(
            f"coord_descent is only supported for SINDy-Attention encoders, not for {args.encoder}"
        )
    if args.coord_descent and args.coord_descent_model_n_epochs <= 0:
        raise ValueError(
            f"coord_descent_model_n_epochs {args.coord_descent_model_n_epochs} must be greater than 0"
        )
    if args.coord_descent and args.coord_descent_sindy_layer_n_epochs <= 0:
        raise ValueError(
            f"coord_descent_sindy_layer_n_epochs {args.coord_descent_sindy_layer_n_epochs} must be greater than 0"
        )
    if args.coord_descent and args.coord_descent_model_lr <= 0:
        raise ValueError(
            f"coord_descent_model_lr {args.coord_descent_model_lr} must be greater than 0"
        )
    if args.coord_descent and args.coord_descent_sindy_layer_lr <= 0:
        raise ValueError(
            f"coord_descent_sindy_layer_lr {args.coord_descent_sindy_layer_lr} must be greater than 0"
        )
    if args.dataset not in ["planetswe", "sst", "plasma", "sst_demo"]:
        raise ValueError(
            f"dataset {args.dataset} not supported, choose one of: planetswe, sst, plasma, sst_demo"
        )
    if args.decoder not in ["mlp", "cnn"]:
        raise ValueError(
            f"decoder {args.decoder} not supported, choose one of: mlp, cnn"
        )
    if args.decoder_depth <= 0:
        raise ValueError(f"decoder_depth {args.decoder_depth} must be greater than 0")
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError(f"dropout {args.dropout} must be between 0 and 1")
    if "sindy_loss" in args.encoder and args.dt < 0:
        raise ValueError(f"dt {args.dt} must be non-negative")
    if args.early_stop < 0:
        raise ValueError(f"early_stop {args.early_stop} must be non-negative")
    if args.encoder not in [
        "gru",
        "lstm",
        "mlp",
        "moe_gru",
        "moe_lstm",
        "moe_mlp",
        "sindy_loss_gru",
        "sindy_loss_lstm",
        "sindy_loss_mlp",
        "sindy_loss_moe_gru",
        "sindy_loss_moe_mlp",
        "sindy_loss_moe_lstm",
        "vanilla_transformer",
        "sindy_loss_transformer",
        "sindy_attention_transformer",
        "sindy_attention_sindy_loss_transformer",
    ]:
        raise ValueError(
            f"encoder {args.encoder} not supported, choose one of: gru, lstm, mlp, moe_gru, moe_lstm, moe_mlp, sindy_loss_gru, sindy_loss_lstm, sindy_loss_mlp, sindy_loss_moe_gru, sindy_loss_moe_mlp, sindy_loss_moe_lstm, vanilla_transformer, sindy_loss_transformer, sindy_attention_transformer, sindy_attention_sindy_loss_transformer"
        )
    if args.encoder_depth <= 0:
        raise ValueError(f"encoder_depth {args.encoder_depth} must be greater than 0")
    if args.epochs <= 0:
        raise ValueError(f"epochs {args.epochs} must be greater than 0")
    if args.forecast_length <= 0:
        raise ValueError(
            f"forecast_length {args.forecast_length} must be greater than 0"
        )
    if args.forecast_length > 1 and args.encoder not in [
        "sindy_attention_transformer",
        "sindy_attention_sindy_loss_transformer",
        "moe_gru",
        "moe_lstm",
        "moe_mlp",
        "sindy_loss_moe_gru",
        "sindy_loss_moe_mlp",
        "sindy_loss_moe_lstm",
    ]:
        raise ValueError(
            f"forecast_length {args.forecast_length} must be 1 for non-rollout encoders"
        )
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size {args.hidden_size} must be greater than 0")
    if args.input_length <= 0:
        raise ValueError(f"input_length {args.input_length} must be greater than 0")
    if args.identifier is None:
        raise ValueError(f"identifier {args.identifier} must be provided")
    if not args.coord_descent and args.lr <= 0:
        raise ValueError(f"lr {args.lr} must be greater than 0")
    if "transformer" in args.encoder and args.n_heads <= 0:
        raise ValueError(f"n_heads {args.n_heads} must be greater than 0")
    if "moe" in args.encoder and args.n_experts <= 0:
        raise ValueError(f"n_experts {args.n_experts} must be greater than 0")
    if args.n_sensors <= 0:
        raise ValueError(f"n_sensors {args.n_sensors} must be greater than 0")
    if args.dataset in ["planetswe"] and args.n_well_tracks <= 0:
        raise ValueError(f"n_well_tracks {args.n_well_tracks} must be greater than 0")
    if "sindy" in args.encoder and args.poly_order <= 0:
        raise ValueError(f"poly_order {args.poly_order} must be greater than 0")
    if args.save_every_n_epochs <= 0:
        raise ValueError(
            f"save_every_n_epochs {args.save_every_n_epochs} must be greater than 0"
        )
    if args.seed < 0:
        raise ValueError(f"seed {args.seed} must be non-negative")
    if "sindy_attention" in args.encoder and args.sindy_layer_threshold < 0:
        raise ValueError(
            f"sindy_layer_threshold {args.sindy_layer_threshold} must be non-negative"
        )
    if "sindy_attention" in args.encoder and args.sindy_layer_threshold_n_epochs <= 0:
        raise ValueError(
            f"sindy_layer_threshold_n_epochs {args.sindy_layer_threshold_n_epochs} must be greater than 0"
        )
    if "sindy_attention" in args.encoder and args.sindy_layer_weight < 0:
        raise ValueError(
            f"sindy_layer_weight {args.sindy_layer_weight} must be non-negative"
        )
    if "sindy_loss" in args.encoder and args.sindy_loss_threshold < 0:
        raise ValueError(
            f"sindy_loss_threshold {args.sindy_loss_threshold} must be non-negative"
        )
    if "sindy_loss" in args.encoder and args.sindy_loss_weight < 0:
        raise ValueError(
            f"sindy_loss_weight {args.sindy_loss_weight} must be non-negative"
        )

    return


def get_dataset_dims(dataset):
    """
    Get the spatial dimensions for a given dataset.

    Args:
        dataset (str): Name of the dataset ("sst" or "planetswe")

    Returns:
        tuple: Tuple of (rows, cols, channels) representing the spatial dimensions

    Raises:
        NotImplementedError: If the dataset name is not recognized
    """
    if dataset == "sst":
        return (180, 360, 1)
    elif dataset == "planetswe":
        return (256, 512, 3)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")


def print_model_size(model, name):
    """
    Print the total memory size of a PyTorch model in megabytes.

    Calculates the combined size of all parameters and buffers in the model.

    Args:
        model (nn.Module): PyTorch model to measure
        name (str): Display name for the model in the output
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 * 1024)
    size_all_mb_before = int(size_all_mb)
    size_all_mb_after = int((size_all_mb - size_all_mb_before) * 100)
    print(f"{name} size: {size_all_mb_before}.{size_all_mb_after:02d}MB")


def print_errors(true_l, pred_l, error_f, title):
    """
    Print error metrics for a list of true and predicted tensor pairs.

    Args:
        true_l (list): List of ground truth tensors
        pred_l (list): List of predicted tensors
        error_f (callable): Error function that takes (true, pred) and returns a scalar
        title (str): Title to print before the error values

    Returns:
        None
    """
    print(title)
    for i, (true, pred) in enumerate(zip(true_l, pred_l)):
        print(f"Error for i={i} is {number_to_percentage(error_f(true, pred))}")
    print()


def mean_absolute_error(datatrue, datapred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted data.

    Args:
        datatrue (torch.Tensor): Ground truth data tensor
        datapred (torch.Tensor): Predicted data tensor

    Returns:
        torch.Tensor: Mean absolute error value
    """
    return (datatrue - datapred).abs().mean()


def mean_squared_error(datatrue, datapred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted data.

    Args:
        datatrue (torch.Tensor): Ground truth data tensor
        datapred (torch.Tensor): Predicted data tensor

    Returns:
        torch.Tensor: Mean squared error value
    """
    return (datatrue - datapred).pow(2).sum(axis=-1).mean()


def mean_relative_error(datatrue, datapred):
    """
    Calculate Mean Relative Error (MRE) between true and predicted data.

    Args:
        datatrue (torch.Tensor): Ground truth data tensor
        datapred (torch.Tensor): Predicted data tensor

    Returns:
        torch.Tensor: Mean relative error value
    """
    return (
        (datatrue - datapred).pow(2).sum(axis=-1).sqrt()
        / (datatrue).pow(2).sum(axis=-1).sqrt()
    ).mean()


def number_to_percentage(prob):
    """
    Convert a decimal probability to a percentage string with 2 decimal places.

    Args:
        prob (float): Probability value between 0 and 1

    Returns:
        str: Formatted percentage string with 2 decimal places and % symbol
    """
    return "%.2f%%" % (100 * prob)


def generate_sensor_positions(
    n_sensors: int, max_rows: int, max_cols: int
) -> list[tuple[int, int]]:
    """
    Generate random sensor positions within a 2D grid.

    Uses a fixed random seed (0) for reproducibility.

    Args:
        n_sensors (int): Number of sensor positions to generate
        max_rows (int): Maximum row index (exclusive)
        max_cols (int): Maximum column index (exclusive)

    Returns:
        list[tuple[int, int]]: List of (row, col) tuples representing sensor positions
    """
    random.seed(0)
    return [
        (random.randint(0, max_rows - 1), random.randint(0, max_cols - 1))
        for _ in range(n_sensors)
    ]


def print_dictionary(hp_dict: dict[str, str], text: str) -> None:
    """
    Print given dictionary

    Args:
        hp_dict (dict): Dictionary to print key and values for
        text (str): Text to print before dictionary

    Returns:
        None
    """
    print(text)
    for key in sorted(hp_dict.keys()):
        print(f"> {key}: {hp_dict[key]}")
    print()

    return None


def normalize_pytorch(tensor, dims, mean=None, std=None, eps=1e-8):
    """
    Normalize a tensor across its channel dimension.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, W, H, C)
        mean (torch.Tensor, optional): Pre-computed mean values for each channel
        std (torch.Tensor, optional): Pre-computed standard deviation values for each channel
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: Normalized tensor of same shape as input
        torch.Tensor: Mean values used for normalization
        torch.Tensor: Standard deviation values used for normalization
    """
    # Calculate mean and std across all dimensions except channel
    if mean is None:
        mean = tensor.mean(dim=dims, keepdim=True)
    if std is None:
        std = tensor.std(dim=dims, keepdim=True)

    # Normalize
    normalized = (tensor - mean) / (std + eps)

    return normalized, mean, std


def inverse_normalize_pytorch(normalized_tensor, mean, std, eps=1e-8):
    """
    Denormalize a tensor that was previously normalized using normalize_channels.

    Args:
        normalized_tensor (torch.Tensor): Normalized tensor of shape (N, W, H, C)
        mean (torch.Tensor): Mean values used for normalization, shape (1, 1, 1, C)
        std (torch.Tensor): Standard deviation values used for normalization, shape (1, 1, 1, C)
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: Denormalized tensor of same shape as input
    """
    # Denormalize
    denormalized = normalized_tensor * (std + eps) + mean

    return denormalized


def evaluate_model(
    model, dl, sensors, metadata, epoch=0, rollout=False, rmse=False, args=None
):
    """
    Evaluate a PyTorch model on a dataset and compute reconstruction loss.

    Args:
        model (nn.Module): PyTorch model to evaluate
        dl (DataLoader): DataLoader containing evaluation data
        sensors (list): List of (row, col) sensor position tuples
        metadata (dict): Dictionary containing scalers and other dataset metadata
        epoch (int): Current epoch number for plot naming (default: 0)
        rollout (bool): If True, evaluate on full forecast rollout; if False, evaluate next-step only (default: False)
        rmse (bool): If True, return RMSE instead of MSE (default: False)
        args (argparse.Namespace): Configuration arguments

    Returns:
        tuple: (reconstruction_loss, sindy_loss) averaged over all batches
    """
    model.to(args.device)
    scalers = metadata["scalers"]
    loss_fn = torch.nn.MSELoss()
    model.eval()
    dl_loss = 0.0
    sindy_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            # Get raw data
            batch[0] = batch[0].to(args.device)
            batch[1] = batch[1].to(args.device)

            # Create inputs and outputs based on model
            inputs = batch[0][:, : args.input_length, :, :, :]

            # If validation, use full rollout, otherwise for test use next step rollout
            if rollout:
                labels = batch[1][:, args.input_length :, :, :, :]
            else:
                labels = batch[1][:, args.input_length : args.input_length + 1, :, :, :]

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:, :, sensor[0], sensor[1], :])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            # n is number of sensors
            input_sensors = einops.rearrange(input_sensors, "b w n d -> b w (n d)")

            # Pass data through model
            output = model(input_sensors)

            # [batch, forecast_length, sequence_length, (rows x cols x dim)]
            outputs = output["output"]
            sindy_loss_batch = output.get("sindy_loss", None)

            # Reshape output
            expected_seq_len = args.input_length if "transformer" in args.encoder else 1
            outputs = einops.rearrange(
                outputs,
                "batch forecast seq_len (rows cols dim) -> batch forecast seq_len rows cols dim",
                batch=batch[0].shape[0],
                forecast=args.forecast_length,
                seq_len=expected_seq_len,
                rows=args.data_rows_out,
                cols=args.data_cols_out,
                dim=args.d_data_out,
            )

            # Take one rollout during test split, otherwise full rollout during validation split
            if rollout:
                outputs = outputs[:, :, -1, :, :, :]
            else:
                outputs = outputs[:, 0:1, -1, :, :, :]

            # Calculate loss
            reconstruction_loss = loss_fn(outputs, labels)

            if sindy_loss_batch is not None:
                sindy_loss_batch = args.sindy_loss_weight * sindy_loss_batch
            if "sindy_attention" in args.encoder or "moe" in args.encoder:
                if args.sindy_layer_weight > 0.0:
                    sindy_sum = (
                        args.sindy_layer_weight
                        * model.get_sindy_layer_coefficients_sum()
                    )

            if rmse:
                reconstruction_loss = torch.sqrt(reconstruction_loss)

            dl_loss += reconstruction_loss.item()

            if sindy_loss_batch is not None:
                sindy_loss += sindy_loss_batch.item()

            # Plot
            if args.generate_training_plots and i == 0:
                if args.dataset != "plasma":
                    outputs = outputs.detach()[0][0]
                    labels = labels[0][0]

                    for j in range(outputs.shape[-1]):
                        outputs[..., j] = inverse_min_max_scale(
                            outputs[..., j], scalers[j]
                        )
                        labels[..., j] = inverse_min_max_scale(
                            labels[..., j], scalers[j]
                        )

                    plot_field_comparison(
                        outputs,
                        labels,
                        dataset=args.dataset,
                        sensors=sensors,
                        save=True,
                        fname=f"{args.identifier}_full_comparison_epoch{epoch}",
                    )
                else:
                    outputs = outputs.detach()[0][0]
                    labels = labels[0][0]

                    # For each feature ...
                    for k in range(14):
                        # Convert from V to full space
                        u = (
                            torch.from_numpy(
                                metadata["u_total"][20 * k : 20 * (k + 1), :]
                            )
                            .float()
                            .to(args.device)
                        )
                        s = (
                            torch.from_numpy(metadata["s_total"][:, k])
                            .float()
                            .to(args.device)
                        )
                        v = (
                            torch.from_numpy(
                                metadata["v_total"][:, 20 * k : 20 * (k + 1)]
                            )
                            .float()
                            .to(args.device)
                        )

                        true_shaped = (
                            labels[0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                        )
                        output_shaped = (
                            outputs[0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                        )

                        true_shaped = einops.rearrange(
                            true_shaped,
                            "(r c) -> c r ()",
                            r=args.data_rows_in,
                            c=args.data_cols_in,
                        )
                        output_shaped = einops.rearrange(
                            output_shaped,
                            "(r c) -> c r ()",
                            r=args.data_rows_in,
                            c=args.data_cols_in,
                        )

                        plot_field_comparison(
                            output_shaped,
                            true_shaped,
                            dataset=args.dataset,
                            sensors=sensors,
                            save=True,
                            fname=f"{args.identifier}_f{k+1}_full_comparison_epoch{epoch}",
                        )

        # Average loss
        dl_loss /= len(dl)
        if sindy_loss_batch is not None:
            sindy_loss /= len(dl)

    return dl_loss, sindy_loss


def create_far_out_plots(model, ds, sensors, metadata, args=None):
    """
    Generate visualization plots for long-horizon forecasting.

    Creates comparison plots between model predictions and ground truth at
    multiple timesteps into the future. For non-rollout models, performs
    autoregressive forecasting.

    Args:
        model (nn.Module): Trained PyTorch model
        ds (Dataset): Dataset to sample from for visualization
        sensors (list): List of (row, col) sensor position tuples
        metadata (dict): Dictionary containing scalers and SVD components
        args (argparse.Namespace): Configuration arguments

    Returns:
        None
    """
    model.eval()

    # Which timesteps to evaluate
    if args.dataset == "plasma":
        ds_iter = [0]
        forecast_length_plot = 101
    elif args.dataset == "planetswe":
        ds_iter = [0]
        forecast_length_plot = 101
    elif args.dataset == "sst":
        ds_iter = [0]
        forecast_length_plot = 101
    plot_steps = list(np.linspace(0, forecast_length_plot - 1, 3).astype(int))

    with torch.no_grad():
        for i in ds_iter:
            # Get raw data
            data = ds[i]

            # Create inputs and outputs based on model, assuming transformer rollout type
            # Transformers are causal (masked self-attention)
            inputs = data[0][: args.input_length, :, :, :].to(args.device)
            labels = data[1][-1, :, :, :].to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:, sensor[0], sensor[1], :])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, "w n d -> 1 w (n d)")

            # Pass data through model
            expected_seq_len = args.input_length if "transformer" in args.encoder else 1
            if "sindy_attention" in args.encoder or "moe" in args.encoder:
                model.encoder.set_forecast_length(forecast_length_plot)
                output = model(input_sensors)
                model.encoder.set_forecast_length(args.forecast_length)
                outputs = output["output"]
            else:
                # Autoregressively forecast since forecast length is 1
                preds = []
                curr_inputs = inputs.clone()
                for j in range(forecast_length_plot):
                    # Build sensors from current input window
                    step_input_sensors = []
                    for sensor in sensors:
                        step_input_sensors.append(
                            curr_inputs[:, sensor[0], sensor[1], :]
                        )
                    step_input_sensors = torch.stack(
                        step_input_sensors, dim=2
                    )  # [w, n, d]
                    step_input_sensors = einops.rearrange(
                        step_input_sensors, "w n d -> 1 w (n d)"
                    )

                    step_output = model(step_input_sensors)
                    preds.append(
                        step_output["output"]
                    )  # shape [1, 1, seq_len, (r c d)]

                    # Extract predicted next frame (last in seq_len)
                    step_output_reshaped = einops.rearrange(
                        step_output["output"],
                        "b f s (r c d) -> b f s r c d",
                        b=1,
                        f=1,
                        s=expected_seq_len,
                        r=args.data_rows_out,
                        c=args.data_cols_out,
                        d=args.d_data_out,
                    )
                    next_frame = step_output_reshaped[0, 0, -1]  # [r, c, d]

                    if args.dataset == "plasma":
                        # Convert from V to full space
                        u = (
                            torch.from_numpy(metadata["u_total"])
                            .float()
                            .to(args.device)
                        )
                        s = (
                            torch.from_numpy(metadata["s_total"])
                            .float()
                            .to(args.device)
                        )
                        v = (
                            torch.from_numpy(metadata["v_total"])
                            .float()
                            .to(args.device)
                        )

                        s_diag = torch.diag_embed(s.reshape(-1))
                        next_frame = next_frame[0, :, 0] @ s_diag @ u

                        # Reshape
                        next_frame = einops.rearrange(
                            next_frame,
                            "(r c d) -> r c d",
                            r=args.data_rows_in,
                            c=args.data_cols_in,
                            d=args.d_data_in,
                        )

                    # Slide window: drop first frame, append prediction
                    curr_inputs = torch.cat(
                        [curr_inputs[1:], next_frame.unsqueeze(0)], dim=0
                    )

                outputs = torch.cat(preds, dim=1)  # [1, forecast, seq_len, (r c d)]

            # Reshape output
            outputs = einops.rearrange(
                outputs,
                "1 forecast seq_len (r c d) -> forecast seq_len r c d",
                forecast=forecast_length_plot,
                seq_len=expected_seq_len,
                r=args.data_rows_out,
                c=args.data_cols_out,
                d=args.d_data_out,
            )

            # Extract only last column of forecast corresponding to full input and predicting all unseen states
            outputs = outputs[:, -1, :, :, :]

            # Convert back to original scale (except for plasma)
            if args.dataset not in ["plasma"]:
                for j in range(outputs.shape[3]):
                    outputs[..., j] = inverse_min_max_scale(
                        outputs[..., j], metadata["scalers"][j]
                    )

                for j in plot_steps:
                    tmp_label = ds[i + args.input_length + j][1][0, :, :, :]
                    tmp_label.to(outputs[j].device)

                    for k in range(outputs.shape[3]):
                        tmp_label[..., k] = inverse_min_max_scale(
                            tmp_label[..., k], metadata["scalers"][k]
                        )

                    plot_field_comparison(
                        outputs[j],
                        tmp_label,
                        dataset=args.dataset,
                        sensors=sensors,
                        save=True,
                        fname=f"{args.identifier}_full_comparison_ds{i}_r{j}",
                    )
            elif args.dataset in ["plasma"]:
                for j in plot_steps:
                    tmp_label = ds[i + args.input_length + j][1][0, :, :, :]
                    tmp_label = tmp_label.to(outputs[j].device)

                    # For each feature ...
                    for k in range(14):
                        # Convert from V to full space
                        u = (
                            torch.from_numpy(
                                metadata["u_total"][20 * k : 20 * (k + 1), :]
                            )
                            .float()
                            .to(args.device)
                        )
                        s = (
                            torch.from_numpy(metadata["s_total"][:, k])
                            .float()
                            .to(args.device)
                        )
                        v = (
                            torch.from_numpy(
                                metadata["v_total"][:, 20 * k : 20 * (k + 1)]
                            )
                            .float()
                            .to(args.device)
                        )

                        true_shaped = (
                            tmp_label[0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                        )
                        output_shaped = (
                            outputs[j][0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                        )

                        true_shaped = einops.rearrange(
                            true_shaped,
                            "(r c) -> c r ()",
                            r=args.data_rows_in,
                            c=args.data_cols_in,
                        )
                        output_shaped = einops.rearrange(
                            output_shaped,
                            "(r c) -> c r ()",
                            r=args.data_rows_in,
                            c=args.data_cols_in,
                        )

                        plot_field_comparison(
                            output_shaped,
                            true_shaped,
                            dataset=args.dataset,
                            sensors=sensors,
                            save=True,
                            fname=f"{args.identifier}_f{k+1}_full_comparison_ds{i}_r{j}",
                        )


def create_next_step_plots(model, ds, sensors, metadata, args=None):
    """
    Generate visualization plots for next-step predictions.

    Creates comparison plots between model predictions and ground truth
    for single-step forecasting at multiple dataset indices.

    Args:
        model (nn.Module): Trained PyTorch model
        ds (Dataset): Dataset to sample from for visualization
        sensors (list): List of (row, col) sensor position tuples
        metadata (dict): Dictionary containing scalers and SVD components
        args (argparse.Namespace): Configuration arguments

    Returns:
        None
    """
    model.eval()

    # Which timesteps to evaluate
    if args.dataset == "plasma":
        ds_iter = [0, 49, 99, 149]
    elif args.dataset == "planetswe":
        ds_iter = [0, 24, 49]
    elif args.dataset == "sst":
        ds_iter = [0, 44, 89]

    with torch.no_grad():
        for i in ds_iter:
            # Get raw data
            data = ds[i]

            # Create inputs and outputs, not causal
            inputs = data[0][: args.input_length, :, :, :].to(args.device)
            labels = data[1][-1, :, :, :].to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:, sensor[0], sensor[1], :])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, "w n d -> 1 w (n d)")

            # Pass data through model
            output = model(input_sensors)

            outputs = output["output"]

            # Reshape output
            expected_seq_len = args.input_length if "transformer" in args.encoder else 1
            outputs = einops.rearrange(
                outputs,
                "1 1 seq_len (r w d) -> seq_len r w d",
                seq_len=expected_seq_len,
                r=args.data_rows_out,
                w=args.data_cols_out,
                d=args.d_data_out,
            )

            outputs = outputs[0]

            # Convert back to original scale (except for plasma)
            if args.dataset not in ["plasma"]:
                for j in range(outputs.shape[2]):
                    outputs[:, :, j] = inverse_min_max_scale(
                        outputs[:, :, j], metadata["scalers"][j]
                    )
                    labels[:, :, j] = inverse_min_max_scale(
                        labels[:, :, j], metadata["scalers"][j]
                    )

                plot_field_comparison(
                    outputs,
                    labels,
                    dataset=args.dataset,
                    sensors=sensors,
                    save=True,
                    fname=f"{args.identifier}_full_comparison_{i}",
                )
            elif args.dataset in ["plasma"]:
                # For each feature ...
                for k in range(14):
                    # Convert from V to full space
                    u = (
                        torch.from_numpy(metadata["u_total"][20 * k : 20 * (k + 1), :])
                        .float()
                        .to(args.device)
                    )
                    s = (
                        torch.from_numpy(metadata["s_total"][:, k])
                        .float()
                        .to(args.device)
                    )
                    v = (
                        torch.from_numpy(metadata["v_total"][:, 20 * k : 20 * (k + 1)])
                        .float()
                        .to(args.device)
                    )

                    true_shaped = (
                        labels[0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                    )
                    output_shaped = (
                        outputs[0, 20 * k : 20 * (k + 1), 0] @ torch.diag(s) @ u
                    )

                    true_shaped = einops.rearrange(
                        true_shaped,
                        "(r c) -> c r ()",
                        r=args.data_rows_in,
                        c=args.data_cols_in,
                    )
                    output_shaped = einops.rearrange(
                        output_shaped,
                        "(r c) -> c r ()",
                        r=args.data_rows_in,
                        c=args.data_cols_in,
                    )

                    plot_field_comparison(
                        output_shaped,
                        true_shaped,
                        dataset=args.dataset,
                        sensors=sensors,
                        save=True,
                        fname=f"{args.identifier}_f{k+1}_full_comparison_{i}",
                    )


def coord_descent_change_lr(optimizer, epoch, args):
    """
    Adjust learning rates for coordinate descent training.

    Alternates between training the main model and SINDy layer coefficients
    by setting one learning rate to zero while the other is active.

    Alternation occurs by checking the current epoch and comparing with how many
    epochs to train each parameter group for.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer with two param groups
            (SINDy coefficients and model parameters)
        epoch (int): Current epoch number
        args (argparse.Namespace): Configuration arguments containing
            coord_descent_model_n_epochs, coord_descent_sindy_layer_n_epochs,
            coord_descent_model_lr, and coord_descent_sindy_layer_lr

    Returns:
        None
    """
    remainder = epoch % (
        args.coord_descent_sindy_layer_n_epochs + args.coord_descent_model_n_epochs
    )
    if remainder < args.coord_descent_model_n_epochs:
        # Train model
        optimizer.param_groups[0]["lr"] = 0.0
        optimizer.param_groups[1]["lr"] = args.coord_descent_model_lr
    else:
        # Train SINDy-Attention coefficients
        optimizer.param_groups[0]["lr"] = args.coord_descent_sindy_layer_lr
        optimizer.param_groups[1]["lr"] = 0.0


def create_inputs_and_labels_from_batch(batch, args):
    """
    Create model inputs and labels from a batch based on encoder type.

    Different encoder architectures require different label formats:
    - Transformers use causal labels (shifted by 1)
    - SINDy-Attention uses stacked multi-step forecasts
    - MOE models use long-term forecast labels
    - RNNs use single next-step labels

    Args:
        batch (list): List of [inputs, outputs] tensors with shape
            (batch_size, input_length + forecast_length, rows, cols, dim)
        args (argparse.Namespace): Configuration arguments

    Returns:
        tuple: (inputs, labels) tensors formatted for the specific encoder type
            inputs: shaped (batch_size, input_length, rows, cols, dim)
            outputs: shaped (batch_size, forecast_length, sequence_length, rows, cols, dim)
                Unless the model is sindy-attention or moe, forecast length is 1.
    """
    # Inputs are the same regardless of model
    inputs = batch[0][:, : args.input_length, :, :, :]

    # Create inputs and outputs from batch based on model
    if "transformer" in args.encoder and "sindy_attention" not in args.encoder:
        # Transformers are causal (masked self-attention)
        labels = batch[1][:, 1:, :, :, :]
    elif "transformer" in args.encoder and "sindy_attention" in args.encoder:
        # SINDy-Attention Transformers create a long-term forecast for each input
        labels = torch.stack(
            [
                batch[1][:, (i + 1) : (i + 1) + args.forecast_length, :, :, :]
                for i in range(args.input_length)
            ],
            dim=2,
        )
    elif "moe" in args.encoder:
        # MOE-GRUs create one long-term forecast
        labels = batch[1][:, args.input_length :, :, :, :]
    else:
        # LSTMs and GRUs are not causal, so we use the last timestep as the label
        # > forecast length is 1
        labels = batch[1][:, -1:, :, :, :]

    if (
        "sindy_attention" not in args.encoder
        and "moe" not in args.encoder
    ):
        # Set forecast length to 1 for non-SINDy-Layer models
        labels = labels.unsqueeze(1)
    elif "moe" in args.encoder:
        # Set sequence length to 1 for moe models
        labels = labels.unsqueeze(2)

    return inputs, labels


def train_model(
    model,
    train_dl,
    val_dl,
    sensors,
    start_epoch,
    best_val,
    best_epoch,
    train_losses,
    val_losses,
    model_eigvs,
    optimizer,
    metadata,
    args,
):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dl (DataLoader): PyTorch DataLoader instance for training data.
        val_dl (DataLoader): PyTorch DataLoader instance for validation data.
        sensors (list): List of sensor locations.
        start_epoch (int): Epoch to start training from.
        best_val (float): Best validation loss.
        best_epoch (int): Epoch of best validation loss.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        model_eigvs (list): List of model eigenvalues (SINDy-Attention Transformer w/ Rollout only).
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        metadata (dict): Dictionary of metadata for the dataset
        args (argparse.Namespace): Arguments to use for training.

    Returns:
        None
    """
    # Set up model, optimizer, and loss
    loss_fn = torch.nn.MSELoss()
    model.to(args.device)
    scalers = metadata["scalers"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Calculate training loss
        train_loss = 0.0
        sindy_loss = 0.0

        if args.coord_descent:
            # Change learning rate based on epoch
            coord_descent_change_lr(optimizer, epoch, args)

        for i, batch in enumerate(train_dl):
            # Get raw data
            batch[0] = batch[0].to(args.device)
            batch[1] = batch[1].to(args.device)

            inputs, labels = create_inputs_and_labels_from_batch(batch, args)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:, :, sensor[0], sensor[1], :])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            # n is number of sensors
            input_sensors = einops.rearrange(input_sensors, "b w n d -> b w (n d)")

            optimizer.zero_grad()

            # Pass data through model
            output = model(input_sensors)

            # [batch, forecast_length, sequence_length, (rows x cols x dim)]
            outputs = output["output"]
            sindy_loss_batch = output.get("sindy_loss", None)

            # Reshape output
            expected_seq_len = args.input_length if "transformer" in args.encoder else 1
            outputs = einops.rearrange(
                outputs,
                "batch forecast seq_len (rows cols dim) -> batch forecast seq_len rows cols dim",
                batch=batch[0].shape[0],
                forecast=args.forecast_length,
                seq_len=expected_seq_len,
                rows=args.data_rows_out,
                cols=args.data_cols_out,
                dim=args.d_data_out,
            )

            # Calculate loss
            reconstruction_loss = loss_fn(outputs, labels)

            # Add other losses if available
            loss = reconstruction_loss
            if sindy_loss_batch is not None:
                sindy_loss_batch = args.sindy_loss_weight * sindy_loss_batch
                loss += sindy_loss_batch
            if "sindy_attention" in args.encoder or "moe" in args.encoder:
                if args.sindy_layer_weight > 0.0:
                    sindy_sum = (
                        args.sindy_layer_weight
                        * model.get_sindy_layer_coefficients_sum()
                    )
                    loss += sindy_sum

            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if sindy_loss_batch is not None:
                sindy_loss += sindy_loss_batch.item()

        # Threshold if necessary
        if "sindy_attention" in args.encoder or "moe" in args.encoder:
            if epoch > 0 and (epoch + 1) % args.sindy_layer_threshold_n_epochs == 0:
                if args.verbose:
                    print(f"Thresholding SINDy coefficients (epoch {epoch+1})")
                model.encoder.threshold_sindy_layer_coefficients(
                    args.sindy_layer_threshold, verbose=args.verbose
                )

        # Average loss
        train_loss /= len(train_dl)
        if sindy_loss_batch is not None:
            sindy_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Calculate validation loss
        val_loss, sindy_val_loss = evaluate_model(
            model,
            val_dl,
            sensors,
            epoch=epoch,
            metadata=metadata,
            rollout=True,
            rmse=False,
            args=args,
        )
        val_losses.append(val_loss)

        # Save model to checkpoint if validation loss is lower than best validation loss
        if epoch > args.early_stop and val_loss < best_val:
            if args.verbose:
                print()
                print(
                    f"Saving model to {args.best_checkpoint_path}, validation loss improved from {best_val:0.4e} to {val_loss:0.4e}, "
                )
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "model_eigvs": model_eigvs,
                    "sensors": sensors,
                },
                args.best_checkpoint_path,
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "model_eigvs": model_eigvs,
                    "sensors": sensors,
                },
                args.latest_checkpoint_path,
            )
            if args.verbose:
                print()

        # Save model to checkpoint if save_every_n_epochs is reached
        if (epoch + 1) % args.save_every_n_epochs == 0:
            if args.verbose:
                print()
                print(f"Saving model to {args.latest_checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "model_eigvs": model_eigvs,
                    "sensors": sensors,
                },
                args.latest_checkpoint_path,
            )
            if args.verbose:
                print()

        # Print loss
        if args.verbose:
            print(
                f"Epoch {epoch+1}, Training loss: {train_loss:0.4e}, Validation loss: {val_loss:0.4e} (best: {best_val:0.4e})"
            )
            if sindy_loss_batch is not None:
                print(
                    f"Epoch {epoch+1}, SINDy training loss: {sindy_loss:0.4e}, SINDy validation loss: {sindy_val_loss:0.4e}"
                )

        # Print model coefficients
        if args.verbose and (
            "sindy_attention" in args.encoder or "moe" in args.encoder
        ):
            model.encoder.print_sindy_layer_coefficients()

        # Collect model eigenvalues
        if "sindy_attention" in args.encoder or "moe" in args.encoder:
            model_eigvs_epoch = model.encoder.get_sindy_layer_coefficients_eigenvalues()
            model_eigvs.append(model_eigvs_epoch)

        # Make plots
        if args.generate_loss_plots:
            plot_losses(
                train_losses,
                val_losses,
                best_epoch,
                save=True,
                fname=f"{args.identifier}_losses",
            )

    if args.verbose:
        print(f"Training complete, best validation loss: {best_val:0.4e}")
        print()


def min_max_scale(tensor, feature_range=(0, 1), scaler=None):
    """
    Scale a tensor to a given feature range using min-max normalization.

    Args:
        tensor (torch.Tensor): Input tensor to be scaled
        feature_range (tuple): Desired range of transformed data (default: (0, 1))
        scaler (tuple): Tuple of (min, max) values used for scaling (for inverse transformation)

    Returns:
        torch.Tensor: Scaled tensor
        tuple: (min, max) values used for scaling (for inverse transformation) or scaler if provided
    """
    # Ensure the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    if scaler is None:
        # Calculate min and max
        t_min = tensor.min()
        t_max = tensor.max()
    else:
        t_min, t_max = scaler

    # Avoid division by zero
    t_range = t_max - t_min
    if t_range == 0:  # all values are the same
        t_range = 1

    # Scale to [0, 1] first
    scaled = (tensor - t_min) / t_range

    # Then scale to feature_range
    min_range, max_range = feature_range
    scaled = scaled * (max_range - min_range) + min_range

    return scaled, (t_min, t_max)


def inverse_min_max_scale(scaled_tensor, original_min_max, feature_range=(0, 1)):
    """
    Inverse transformation of min-max scaling.

    Args:
        scaled_tensor (torch.Tensor): Scaled tensor to transform back
        original_min_max (tuple): (min, max) values from original scaling
        feature_range (tuple): Range used in original scaling (default: (0, 1))

    Returns:
        torch.Tensor: Tensor in original scale
    """
    t_min, t_max = original_min_max
    min_range, max_range = feature_range

    # First scale back to [0, 1] range
    normalized = (scaled_tensor - min_range) / (max_range - min_range)

    # Then scale back to original range
    original = normalized * (t_max - t_min) + t_min

    return original


def create_mats_full(train, valid, test, total_tracks, debug=False):
    """
    Concatenate data from train, valid, and test splits of The Well into a single matrix.
    Used in preprocess_the_well.py to create a single dataset matrix.

    Collects data from training first, then validation, then test until
    the desired number of tracks is reached.

    Args:
        train (list): List of training data dictionaries with "input_fields" key
        valid (list): List of validation data dictionaries with "input_fields" key
        test (list): List of test data dictionaries with "input_fields" key
        total_tracks (int): Maximum number of tracks to include
        debug (bool): If True, stop after first track (default: False)

    Returns:
        torch.Tensor: Concatenated tensor of all input fields
    """
    im_shape = train[0]["input_fields"].shape
    n_steps, im_rows, im_cols, im_dim = (
        im_shape[0],
        im_shape[1],
        im_shape[2],
        im_shape[3],
    )

    track_count = 0

    mats = []
    for i in range(len(train)):
        # data = einops.rearrange(train[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
        data = train[i]["input_fields"]
        mats.append(data)
        track_count += 1
        if track_count >= total_tracks:
            break
        if debug:
            break
    if track_count < total_tracks:
        for i in range(len(valid)):
            # data = einops.rearrange(valid[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
            data = valid[i]["input_fields"]
            mats.append(data)
            track_count += 1
            if track_count >= total_tracks:
                break
            if debug:
                break
    if track_count < total_tracks:
        for i in range(len(test)):
            # data = einops.rearrange(test[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
            data = test[i]["input_fields"]
            mats.append(data)
            track_count += 1
            if track_count >= total_tracks:
                break
            if debug:
                break
    mats = torch.cat(mats, dim=0)
    return mats


def create_mats(the_well_data, combine_all=False, debug=False):
    """
    Extract and flatten input fields from The Well dataset.

    Args:
        the_well_data (list): List of data dictionaries with "input_fields" key
        combine_all (bool): If True, concatenate all tracks into a single tensor (default: False)
        debug (bool): If True, stop after first track (default: False)

    Returns:
        list: List of flattened tensors of shape (timesteps, rows*cols*dim),
            or single-element list if combine_all is True
    """
    im_shape = the_well_data[0]["input_fields"].shape
    n_steps, im_rows, im_cols, im_dim = (
        im_shape[0],
        im_shape[1],
        im_shape[2],
        im_shape[3],
    )

    mats = []
    for i in range(len(the_well_data)):
        data = einops.rearrange(
            the_well_data[i]["input_fields"],
            "t r c d -> t (r c d)",
            t=n_steps,
            r=im_rows,
            c=im_cols,
            d=im_dim,
        )
        mats.append(data)
        if debug:
            break
    if combine_all:
        mats = [torch.cat(mats, dim=0)]
    return mats


def generate_SVD(mat, n_rank=50, n_iters=2):
    """
    Compute low-rank SVD decomposition of a matrix.

    Args:
        mat (torch.Tensor): Input matrix to decompose
        n_rank (int): Number of singular values/vectors to compute (default: 50)
        n_iters (int): Number of iterations for the algorithm (default: 2)

    Returns:
        tuple: (U, S, V) matrices from the SVD decomposition
    """
    U, S, V = torch.svd_lowrank(mat, n_rank, n_iters)
    return U, S, V


def create_pod(mat, V):
    """
    Project a matrix onto POD (Proper Orthogonal Decomposition) basis.

    Args:
        mat (torch.Tensor): Input matrix to project
        V (torch.Tensor): Right singular vectors (POD modes)

    Returns:
        torch.Tensor: POD coefficients (reduced representation)
    """
    pod = mat @ V
    return pod


def scale_pod(pod):
    """
    Apply min-max scaling to POD coefficients.

    Args:
        pod (torch.Tensor): POD coefficients to scale

    Returns:
        tuple: (scaled_pod, scalers) where scalers is (min, max) tuple
    """
    pod_scaled, scalers = min_max_scale(pod)
    return pod_scaled, scalers


def inverse_pods_torch(pods_scaled, scalers, V, device=None):
    """
    Reconstruct original data from scaled POD coefficients (batched, PyTorch).

    Args:
        pods_scaled (torch.Tensor): Batch of scaled POD coefficients
        scalers (tuple): (min, max) values from original scaling
        V (torch.Tensor): Right singular vectors (POD modes)
        device (str, optional): Device to move tensors to

    Returns:
        torch.Tensor: Batch of reconstructed matrices
    """
    mat_hats = []
    pods_scaled = pods_scaled.to(device)
    V = V.to(device)
    for i in range(pods_scaled.shape[0]):
        pod_scaled = pods_scaled[i]
        mat_hat = inverse_min_max_scale(pod_scaled, scalers)
        mat_hat = mat_hat @ V.T
        mat_hats.append(mat_hat)
    mat_hats = torch.stack(mat_hats, dim=0)
    return mat_hats


def inverse_pods(pods_scaled, scalers, V):
    """
    Reconstruct original data from scaled POD coefficients (list version).

    Args:
        pods_scaled (list): List of scaled POD coefficient tensors
        scalers (tuple): (min, max) values from original scaling
        V (torch.Tensor): Right singular vectors (POD modes)

    Returns:
        list: List of reconstructed matrices
    """
    mat_hats = []
    for pod_scaled in pods_scaled:
        mat_hat = inverse_min_max_scale(pod_scaled, scalers)
        mat_hat = mat_hat @ V.T
        mat_hats.append(mat_hat)
    return mat_hats


def inverse_pod(pod_scaled, scalers, V):
    """
    Reconstruct original data from a single scaled POD coefficient tensor.

    Args:
        pod_scaled (torch.Tensor): Scaled POD coefficients
        scalers (tuple): (min, max) values from original scaling
        V (torch.Tensor): Right singular vectors (POD modes)

    Returns:
        torch.Tensor: Reconstructed matrix
    """
    mat_hat = inverse_min_max_scale(pod_scaled, scalers)
    mat_hat = mat_hat @ V.T
    return mat_hat


def split_mats(data_list):
    """
    Given a list of data, each element being an individual track, where
    each track contains T timesteps and dimension D, extract the first 80% of the timesteps for training,
    the next 10% for validation, and the last 10% for testing per data. Returns a list of training, validation, and testing data.

    Args:
        data_list (list): List of data tracks, where each track is a tensor of shape (T, D) containing T timesteps and D dimensions

    Returns:
        tuple: (train_data, val_data, test_data) where each is a list of tensors containing the respective splits
    """
    train_data = []
    val_data = []
    test_data = []

    for i, data in enumerate(data_list):
        # Calculate split indices
        n_timesteps = data.shape[0]
        train_end = int(0.8 * n_timesteps)
        val_end = int(0.9 * n_timesteps)

        # Split the data
        train_data.append(data[:train_end])
        val_data.append(data[train_end:val_end])
        test_data.append(data[val_end:])

    return train_data, val_data, test_data


def get_results_from_hyper_opt(hyper_opt_dir):
    """
    Load hyperparameter optimization results from YAML files.

    Searches the given directory for tuning history YAML files and returns
    their contents as a list of dictionaries.

    Args:
        hyper_opt_dir (str or Path): Path to the results directory containing tuning folders

    Returns:
        list: List of dictionaries containing tuning results with file_path added
    """
    results = []
    for tune_folder in Path(hyper_opt_dir).iterdir():
        if tune_folder.is_dir():
            for tune_file in tune_folder.iterdir():
                if (
                    tune_file.is_file()
                    and tune_file.name.startswith("tuning_history_")
                    and tune_file.name.endswith(".yaml")
                ):
                    with open(tune_file, "r") as f:
                        data = yaml.safe_load(f)
                        data["file_path"] = tune_file

                        # Modify encoder if rollout is not 1
                        if (
                            "sindy_attention"
                            in data["final_config"]["model"]["encoder"]
                        ):
                            if data["final_config"]["model"]["forecast_length"] != 1:
                                data["final_config"]["model"]["encoder"] = (
                                    data["final_config"]["model"]["encoder"]
                                    + f"_{data['final_config']['model']['forecast_length']}"
                                )

                        results.append(data)
    return results


def get_dictionaries_from_pickles(pickle_dir):
    """
    Returns a list of dictionaries from all the pickles in the given directory.

    Args:
        pickle_dir (str): Path to the pickles directory.
        early_stop (int): If not None, ensure best validation epoch is at least this value.

    Returns:
        List of dictionaries.
    """
    results = []
    for fname in os.listdir(pickle_dir):
        fpath = os.path.join(pickle_dir, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            data["file_path"] = fpath

            # Modify encoder if rollout is not 1
            if "sindy_attention" in data["hyperparameters"]["encoder"]:
                if data["hyperparameters"]["forecast_length"] != 1:
                    data["hyperparameters"]["encoder"] = (
                        data["hyperparameters"]["encoder"]
                        + f"_{data['hyperparameters']['forecast_length']}"
                    )

            results.append(data)
    return results


def get_top_N_models_by_loss(
    results, dataset_name, N=5, encoders=None, result_type="hyper_opt"
):
    """
    Returns the top N models with the lowest test loss for a given dataset.

    Args:
        results (list): List of dictionaries containing the results.
        dataset_name (str): The dataset name to filter by (e.g., 'sst').
        N (int): Number of results to return.
        encoders (list): List of encoders to filter by.
        result_type (str): The type of results to return (e.g., 'hyper_opt' or 'pickle').

    Returns:
        List of tuples: [(filename, loss), ...] sorted by lowest loss.
    """
    # Filter results by dataset
    if result_type == "hyper_opt":
        filtered_results = [
            r for r in results if r["final_config"]["model"]["dataset"] == dataset_name
        ]
        if encoders is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r["final_config"]["model"]["encoder"] in encoders
            ]
        filtered_results.sort(key=lambda x: x["best_value"], reverse=False)
    elif result_type == "pickle":
        filtered_results = [
            r for r in results if r["hyperparameters"]["dataset"] == dataset_name
        ]
        if encoders is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r["hyperparameters"]["encoder"] in encoders
            ]
        filtered_results.sort(key=lambda x: x["test_loss_next"], reverse=False)
    else:
        raise Exception("Invalid result_type:", result_type)

    return filtered_results[:N]


def print_top_N_results(
    results, dataset_name, N=5, encoders=None, result_type="hyper_opt"
):
    """
    Print the top N models with lowest test loss for a dataset.

    Args:
        results (list): List of result dictionaries
        dataset_name (str): Dataset name to filter by
        N (int): Number of top results to print (default: 5)
        encoders (list, optional): List of encoder names to filter by
        result_type (str): Either "hyper_opt" or "pickle" (default: "hyper_opt")

    Returns:
        None
    """
    results = get_top_N_models_by_loss(results, dataset_name, N, encoders, result_type)
    print(f"{dataset_name} ({N} best)")
    for result in results:
        if result_type == "hyper_opt":
            print(
                f"> Encoder (n={result['final_config']['model']['encoder_depth']}): {result['final_config']['model']['encoder']}"
            )
            print(
                f"> Decoder (n={result['final_config']['model']['decoder_depth']}): {result['final_config']['model']['decoder']}"
            )
            print(f"> Test loss: {result['best_value']:0.4e}")
            print(f"> Config path: {result['file_path']}")
        elif result_type == "pickle":
            print(
                f"> Encoder (n={result['hyperparameters']['encoder_depth']}): {result['hyperparameters']['encoder']}"
            )
            print(
                f"> Decoder (n={result['hyperparameters']['decoder_depth']}): {result['hyperparameters']['decoder']}"
            )
            print(f"> Test loss: {result['test_loss_next']:0.4e}")
            print(f"> Config path: {result['file_path']}")
        else:
            raise Exception("Invalid result_type:", result_type)
        print()


def get_identifier(filename):
    """
    Extract identifier from filename by removing extension and _test_loss suffix.

    Args:
        filename (str or Path): Filename to extract identifier from

    Returns:
        str: Extracted identifier string
    """
    name = Path(filename).stem  # Remove extension
    if name.endswith("_test_loss"):
        name = name[:-10]  # Remove _test_loss suffix
    return name


def generate_sinusoid_sum(n_sin: int, X: int, T: int, seed: int = 42) -> torch.Tensor:
    """
    Generate time series data by summing multiple sinusoids with random frequencies and amplitudes.

    Args:
        n_sin (int): Number of sinusoids to sum
        X (int): Number of time series to generate
        T (int): Number of time steps per series
        seed (int): Random seed for reproducibility

    Returns:
        torch.Tensor: Generated time series of shape (X, T)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Create time points
    t = torch.linspace(0, 2 * torch.pi, T)

    # Initialize output tensor
    output = torch.zeros((X, T))

    # Generate each time series
    for i in range(X):
        # Sum all sinusoids
        for j in range(n_sin):
            # Generate random frequencies and amplitudes
            frequencies = (
                torch.rand(n_sin) * 4 * torch.pi
            )  # Random frequencies between 0 and 2π
            amplitudes = torch.rand(n_sin) * 2  # Random amplitudes between 0 and 2
            output[i] += amplitudes[j] * torch.sin(frequencies[j] * t)

    return output


def extract_config_value(config_file, key):
    """
    Extract a value from a YAML configuration file.

    Args:
        config_file (str or Path): Path to the YAML config file
        key (str): Key to extract from the "model" section

    Returns:
        str: String value of the extracted key
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return str(config["model"][key])


def sort_bash_config_key(config_file):
    """
    Generate a sort key for ordering configuration files.

    Orders by: seed, dataset priority, encoder priority, decoder priority.

    Args:
        config_file (str or Path): Path to the YAML config file

    Returns:
        tuple: Sort key tuple (seed, dataset_priority, encoder_priority, decoder_priority)
    """
    dataset = extract_config_value(config_file, "dataset")
    seed = extract_config_value(config_file, "seed")
    encoder = extract_config_value(config_file, "encoder")
    decoder = extract_config_value(config_file, "decoder")
    # Define priority orders
    dataset_priority = {"planetswe": 0, "sst": 1, "plasma": 2}
    encoder_priority = {
        "gru": 0,
        "lstm": 1,
        "sindy_loss_gru": 2,
        "sindy_loss_lstm": 3,
        "vanilla_transformer": 4,
        "sindy_loss_transformer": 5,
        "sindy_attention_transformer": 6,
        "sindy_attention_sindy_loss_transformer": 7,
    }
    decoder_priority = {"mlp": 0, "cnn": 1}

    # Return tuple for sorting
    return (
        int(seed),
        dataset_priority.get(dataset, 999),
        encoder_priority.get(encoder, 999),
        decoder_priority.get(decoder, 999),
    )


def create_results_table(results, datasets, results_type):
    """
    Create a LaTeX table summarizing model performance across datasets.

    Generates a formatted LaTeX table showing mean ± std RMSE for each
    encoder/decoder combination across specified datasets.

    Args:
        results (list): List of result dictionaries
        datasets (list): List of dataset names to include in the table
        results_type (str): Either "hyper_opt" or "pickle"

    Returns:
        str: LaTeX table code as a string
    """
    import numpy as np

    encoder_order = [
        "lstm",
        "gru",
        "sindy_loss_gru",
        "sindy_loss_lstm",
        "vanilla_transformer",
        "sindy_loss_transformer",
        "sindy_attention_transformer",
        "sindy_attention_sindy_loss_transformer",
        "sindy_attention_transformer_5",
        "sindy_attention_sindy_loss_transformer_5",
    ]
    encoder_label = {
        "lstm": "LSTM",
        "gru": "GRU",
        "sindy_loss_gru": "SL-GRU",
        "sindy_loss_lstm": "SL-LSTM",
        "vanilla_transformer": "T",
        "sindy_loss_transformer": "SL-T",
        "sindy_attention_transformer": "SA-T",
        "sindy_attention_sindy_loss_transformer": "SASL-T",
        "sindy_attention_transformer_5": "SA-T-5",
        "sindy_attention_sindy_loss_transformer_5": "SASL-T-5",
    }
    decoder_order = ["mlp", "cnn"]
    decoder_label = {"mlp": "MLP", "cnn": "CNN"}
    dataset_label = {"plasma": "Plasma", "sst": "SST", "planetswe": "PlanetSWE"}

    from collections import defaultdict

    stats = defaultdict(list)  # (decoder, encoder, dataset) -> [losses]

    for r in results:
        if results_type == "hyper_opt":
            cfg = r["final_config"]["model"]
            ds = cfg["dataset"]
            enc = cfg["encoder"]
            dec = cfg["decoder"]
            loss = r["best_value"]
        elif results_type == "pickle":
            cfg = r["hyperparameters"]
            ds = cfg["dataset"]
            enc = cfg["encoder"]
            dec = cfg["decoder"]
            loss = r["test_loss_next"]
        else:
            raise Exception("Invalid results_type:", results_type)

        if (
            ds not in datasets
            or enc not in encoder_order
            or dec not in decoder_order
            or loss is None
        ):
            continue
        stats[(dec, enc, ds)].append(loss)

    means, stds = {}, {}
    best_per_dataset = {ds: np.inf for ds in datasets}
    for key, losses in stats.items():
        dec, enc, ds = key
        m = float(np.mean(losses))
        s = float(np.std(losses)) if len(losses) > 1 else 0.0
        means[key] = m
        stds[key] = s
        if m < best_per_dataset[ds]:
            best_per_dataset[ds] = m

    def format_cell(decoder, encoder, ds):
        key = (decoder, encoder, ds)
        if key not in means:
            return "—"
        m = means[key]
        s = stds[key]
        cell = f"{m:0.2e} $\\pm$ {s:0.2e}"
        if (
            np.isfinite(best_per_dataset.get(ds, np.inf))
            and abs(m - best_per_dataset[ds]) < 1e-15
        ):
            cell = f"\\textbf{{{cell}}}"
        return cell

    ordered_datasets = [ds for ds in ["plasma", "sst", "planetswe"] if ds in datasets]
    header_cols = [dataset_label.get(ds, ds) for ds in ordered_datasets]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering")
    lines.append(
        f"  \\begin{{tabular}}{{|l|l|{'|'.join(['c']*len(ordered_datasets))}|}}"
    )
    lines.append("  \\hline")
    lines.append(
        "  \\textbf{Decoder} & \\textbf{Encoder} & "
        + " & ".join([f"\\textbf{{{c}}}" for c in header_cols])
        + " \\\\ \\hline"
    )

    total_cols_end = 1 + 1 + len(ordered_datasets)
    for decoder in decoder_order:
        encs_present = [e for e in encoder_order]
        if len(encs_present) == 0:
            continue
        first_enc = encs_present[0]
        lines.append(
            f"  \\multirow{{{len(encs_present)}}}{{*}}{{{decoder_label[decoder]}}} "
            + " & "
            + encoder_label[first_enc]
            + " & "
            + " & ".join(
                [format_cell(decoder, first_enc, ds) for ds in ordered_datasets]
            )
            + f" \\\\ \\cline{{2-{total_cols_end}}}"
        )

        for j, enc in enumerate(encs_present[1:]):
            is_last = j == len(encs_present[1:]) - 1
            row = (
                "              & "
                + encoder_label[enc]
                + " & "
                + " & ".join([format_cell(decoder, enc, ds) for ds in ordered_datasets])
            )
            if is_last:
                row += " \\\\ \\hline"
            else:
                row += f" \\\\ \\cline{{2-{total_cols_end}}}"
            lines.append(row)

    lines.append("  \\end{tabular}")
    lines.append(
        "  \\caption{Model performance across datasets. Values show RMSE of the next-step prediction over five seeds (mean $\\pm$ std). Bold values indicate the best performing model for each dataset. The encoders shown are the vanilla transformer (T), the SINDy-Loss transformer (SL-T), the SINDy-Attention Transformer (SA-T), the SINDy-Attention Transformer with SINDy-Loss (SASL-T), the GRU, the LSTM, the SINDy-Loss GRU (SL-GRU), and the SINDy-Loss LSTM (SL-LSTM).}"
    )
    lines.append("  \\label{tab:results_table}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def execute_command(config_file, sem_dict):
    """
    Execute a training command on a remote machine via SSH.

    Connects to a remote host and executes a training script with the given
    configuration. Logs output to a file.

    Args:
        config_file (str or Path): Path to the YAML config file
        sem_dict (dict): Dictionary containing remote execution settings including:
            - remote_cmd_template: Command template string
            - device: GPU device string (e.g., "cuda:0")
            - log_path: Path for log files
            - computer_name: SSH host name
            - repo_path: Path to repository on remote
            - venv_path: Path to virtual environment on remote

    Returns:
        None
    """
    identifier = extract_config_value(config_file, "identifier")

    remote_cmd_template = sem_dict["remote_cmd_template"]
    command_type = sem_dict["type"]
    device = sem_dict["device"]
    log_path = sem_dict["log_path"]
    computer_name = sem_dict["computer_name"]
    repo_path = sem_dict["repo_path"]
    venv_path = sem_dict["venv_path"]

    device_num = device.split(":")[1]

    remote_cmd = remote_cmd_template.format(
        repo_path=repo_path,
        venv_path=venv_path,
        device_num=device_num,
        identifier=identifier,
        config_file=config_file,
    )

    try:
        # Create logs directory
        log_path.mkdir(exist_ok=True)

        log_filename = f"{identifier}.log"
        log_file = log_path / log_filename

        print(f"Starting remote job: {identifier} on {computer_name}:{device_num}")

        # Execute command remotely via Paramiko SSH
        with open(log_file, "w") as f:
            f.write(
                f"Starting remote job: {identifier} on {computer_name}:{device_num}\n"
            )
            f.write(f"Remote command: {remote_cmd}\n")
            f.flush()

            # Create SSH client and load SSH config
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Load SSH config
            ssh_config = paramiko.SSHConfig()
            ssh_config_path = os.path.expanduser("/home/alexey/.ssh/config")

            if os.path.exists(ssh_config_path):
                with open(ssh_config_path, "r") as config_file:
                    ssh_config.parse(config_file)

            # Get connection details from SSH config
            host_config = ssh_config.lookup(computer_name)

            # Extract connection parameters
            hostname = host_config.get("hostname", computer_name)
            port = int(host_config.get("port", 22))
            username = host_config.get("user", os.getenv("USER", "alexey"))
            key_filename = host_config.get("identityfile")

            # Debug logging
            f.write(
                f"SSH Config - Hostname: {hostname}, Port: {port}, User: {username}\n"
            )
            f.flush()

            try:
                # Connect to remote host using SSH config
                ssh_client.connect(
                    hostname=hostname,
                    port=port,
                    username=username,
                    key_filename=key_filename,
                    timeout=30,
                    allow_agent=True,
                    look_for_keys=True,
                )

                # Execute remote command
                _, stdout, stderr = ssh_client.exec_command(remote_cmd)

                # Read output in real-time and write to log
                exit_status = (
                    stdout.channel.recv_exit_status()
                )  # Wait for command to complete

                # Get all output
                stdout_data = stdout.read().decode("utf-8")
                stderr_data = stderr.read().decode("utf-8")

                # Write output to log
                f.write(stdout_data)
                if stderr_data:
                    f.write(f"\nSTDERR:\n{stderr_data}")

                f.write(
                    f"\nCompleted job: {identifier} on {computer_name}:{device_num} (exit status: {exit_status})\n"
                )

            except Exception as ssh_error:
                error_msg = f"SSH connection error: {ssh_error}"
                f.write(f"\nSSH Error: {error_msg}\n")
                print(f"SSH error for {identifier} on {computer_name}: {ssh_error}")

            finally:
                ssh_client.close()

        print(f"Completed remote job: {identifier} on {computer_name}:{device_num}")

    except Exception as e:
        print(f"Error executing job {identifier} on {computer_name}:{device_num}: {e}")


def worker_thread(config_queue, semaphores):
    """
    Worker thread that processes configuration files from a queue.

    Acquires a semaphore from the available pool and executes commands
    until the queue is empty.

    Args:
        config_queue (Queue): Thread-safe queue of configuration file paths
        semaphores (list): List of semaphore dictionaries for different compute resources

    Returns:
        None
    """
    while True:
        try:
            # Loop through all semaphores, there should be one available since there are not more workers than available semaphores
            semaphore = None
            for sem_dict in semaphores:
                semaphore = sem_dict["semaphore"]
                if semaphore.acquire(blocking=False):
                    break

            if semaphore is None:
                break

            config_file = config_queue.get(timeout=1)

            execute_command(config_file, sem_dict)

            config_queue.task_done()

        except Empty:
            break
        except Exception as e:
            print(f"Worker thread error: {e}")
            break
        finally:
            if semaphore is not None:
                semaphore.release()


def get_tuning_configs(top_dir):
    """
    Find tuning configuration files that haven't been processed yet.

    Searches for tuning_config YAML files and filters out those that
    already have corresponding optimal_params results.

    Args:
        top_dir (Path): Top-level project directory

    Returns:
        list: List of Path objects for unprocessed tuning config files
    """
    configs_dir = top_dir / "configs"
    config_files = []
    for config_file in configs_dir.glob("**/*.yaml"):
        if "tuning_config" in str(config_file):
            config_files.append(config_file)

    config_files.sort(key=sort_bash_config_key)

    config_files_to_process = []
    for config_file in config_files:
        identifier = extract_config_value(config_file, "identifier")
        dataset = extract_config_value(config_file, "dataset")
        results_path = Path("/") / "home" / "alexey" / "Git" / "T-SHRED" / "results"
        results_path = results_path / identifier / f"optimal_params_{dataset}.yaml"

        if not results_path.exists():
            config_files_to_process.append(config_file)

    return config_files_to_process


def get_testing_configs(top_dir):
    """
    Find optimal parameter configs that haven't been tested yet.

    Searches for optimal_params YAML files and filters out those that
    already have corresponding pickle result files.

    Args:
        top_dir (Path): Top-level project directory

    Returns:
        list: List of Path objects for unprocessed optimal params config files
    """
    configs_dir = top_dir / "results"
    config_files = []
    for config_file in configs_dir.glob("**/*.yaml"):
        if "optimal_params" in str(config_file):
            config_files.append(config_file)

    config_files.sort(key=sort_bash_config_key)

    config_files_to_process = []
    for config_file in config_files:
        identifier = extract_config_value(config_file, "identifier")
        results_path = Path("/") / "home" / "alexey" / "Git" / "T-SHRED" / "pickles"
        results_path = results_path / f"{identifier}.pkl"

        if not results_path.exists():
            config_files_to_process.append(config_file)

    return config_files_to_process


def create_all_devices(computers):
    """
    Build a flat list of all available GPU devices across all computers.

    Args:
        computers (dict): Dictionary mapping computer names to their configs,
            where each config has a "gpus" key with a list of GPU device strings

    Returns:
        list: List of (computer_name, gpu_device) tuples
    """
    all_devices = []
    for computer_name, computer_config in computers.items():
        for gpu in computer_config["gpus"]:
            all_devices.append((computer_name, gpu))

    return all_devices


def create_semaphores(computers, n_parallel, remote_cmd_template, command_type):
    """
    Create semaphore dictionaries for parallel job execution.

    Each semaphore controls access to a specific computer-GPU combination,
    limiting concurrent jobs on that resource.

    Args:
        computers (dict): Dictionary mapping computer names to their configs
        n_parallel (int): Maximum number of parallel jobs per GPU
        remote_cmd_template (str): Command template for remote execution
        command_type (str): Type of command (e.g., "train", "test")

    Returns:
        list: List of semaphore dictionaries with execution settings
    """
    semaphores = []
    for computer_name, computer_config in computers.items():
        for gpu in computer_config["gpus"]:
            semaphores.append(
                {
                    "computer_name": computer_name,
                    "device": gpu,
                    "semaphore": threading.Semaphore(n_parallel),
                    "remote_cmd_template": remote_cmd_template,
                    "log_path": Path(computer_config["log_path"]),
                    "repo_path": Path(computer_config["repo_path"]),
                    "venv_path": Path(computer_config["venv_path"]),
                    "type": command_type,
                }
            )

    return semaphores


def run_in_parallel(config_files, semaphores, n_parallel):
    """
    Execute configuration files in parallel using threaded workers.

    Distributes configuration files across available compute resources
    using semaphores to limit concurrent jobs per GPU.

    Args:
        config_files (list): List of configuration file paths to process
        semaphores (list): List of semaphore dictionaries from create_semaphores()
        n_parallel (int): Maximum number of parallel jobs per GPU

    Returns:
        None
    """
    print(f"\nStarting threaded execution of {len(config_files)} configurations...")
    print(
        f"Commands will be distributed across available devices with {n_parallel} jobs per GPU"
    )

    # Create command queue and add all commands
    config_queue = Queue()
    for config_file in config_files:
        config_queue.put(config_file)

    # Create and start worker threads (one per total available slot across all semaphores)
    num_workers = min(
        len(config_files), n_parallel * len(semaphores)
    )  # Don't create more workers than available semaphores
    workers = []

    print(f"Starting {num_workers} worker threads...")

    for i in range(num_workers):
        worker = threading.Thread(
            target=worker_thread, args=(config_queue, semaphores), name=f"Worker-{i}"
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Wait for all commands to complete
    start_time = time.time()
    config_queue.join()

    for worker in workers:
        worker.join()

    end_time = time.time()
    print(f"All commands completed in {end_time - start_time:.2f} seconds!")


def _get_clones(module, N):
    """
    Create N deep copies of a PyTorch module.

    Used for exact parity with PyTorch's transformer implementation.
    Having the same init for every layer might not be strictly necessary.

    Args:
        module (nn.Module): PyTorch module to clone
        N (int): Number of clones to create

    Returns:
        nn.ModuleList: List of N independent copies of the module
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
