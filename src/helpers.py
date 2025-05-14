import torch
import einops
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.plots import plot_losses, plot_field_comparison

def print_errors(true_l, pred_l, error_f, title):
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
    return ((datatrue - datapred).pow(2).sum(axis=-1).sqrt() / (datatrue).pow(2).sum(axis=-1).sqrt()).mean()

def number_to_percentage(prob):
    """
    Convert a decimal probability to a percentage string with 2 decimal places.

    Args:
        prob (float): Probability value between 0 and 1

    Returns:
        str: Formatted percentage string with 2 decimal places and % symbol
    """
    return "%.2f%%" % (100 * prob)


def generate_sensor_positions(n_sensors: int, max_rows: int, max_cols: int) -> list[tuple[int, int]]:
    random.seed(0)
    return [(random.randint(0, max_rows-1), random.randint(0, max_cols-1)) for _ in range(n_sensors)]

def print_dictionary(hp_dict: dict[str, str], text: str) -> None:
    """
    Print given dictionary

    `hp_dict`: dictionary dictionary to print key and values for
    `text`: text to print before dictionary

    Returns: `None`
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

def evaluate_model_pod(model, test_dl, test_full_dl, V, scaler, im_dims, sensors, args):
    """
    Evaluate a PyTorch model.
    """
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    model.eval()
    test_loss_pod = 0.0
    test_loss_pod_full = 0.0
    test_loss_full = 0.0
    with torch.no_grad():
        # Set up iterators for dual dataset loading
        if args.eval_full:
            full_iterator = iter(test_full_dl)
        pod_iterator = iter(test_dl)
        num_iters = len(test_dl)
        for i in range(num_iters):
            # Get batch
            if args.eval_full:
                full_batch = next(full_iterator)
            pod_batch = next(pod_iterator)

            # Get data
            pod_inputs, pod_labels = pod_batch["input_fields"], pod_batch["output_fields"][:,0,:,:,:]
            pod_inputs, pod_labels = pod_inputs.to(args.device), pod_labels.to(args.device)

            if args.eval_full:
                full_inputs, full_labels = full_batch["input_fields"], full_batch["output_fields"][:,0,:]
                full_inputs, full_labels = full_inputs.to(args.device), full_labels.to(args.device)

            # Get real batch size
            batch_size = pod_inputs.shape[0]
            
            # Extract sensors per input tensor
            pod_input_sensors = []
            for sensor in sensors:
                pod_input_sensors.append(pod_inputs[:,:,sensor[0],sensor[1],:])
            pod_input_sensors = torch.stack(pod_input_sensors, dim=2)

            # Prepare input for model
            pod_input_sensors = einops.rearrange(pod_input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            pod_outputs = model(pod_input_sensors)

            # Reshape output (will be: batch x 1 x dim x 1)
            pod_outputs = einops.rearrange(pod_outputs, 'b (r w d) -> b r w d', b=batch_size, r=args.data_rows, w=args.data_cols, d=args.d_data)
            
            # Remove singular dimensions
            pod_outputs_squeezed = pod_outputs[:,0,:,0]
            pod_labels_squeezed = pod_labels[:,0,:,0]

            # Inverse POD to get full scale image
            pod_outputs_full = inverse_pods_torch(pod_outputs_squeezed, scaler, V, device=args.device)
            pod_labels_full = inverse_pods_torch(pod_labels_squeezed, scaler, V, device=args.device)

            # Convert back to original shape
            if args.eval_full:
                pod_outputs_shaped = einops.rearrange(pod_outputs_full, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])
                pod_labels_shaped = einops.rearrange(pod_labels_full, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])
                full_labels_shaped = einops.rearrange(full_labels, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])

            # Generate plots
            if args.eval_full and i == 0:
                plot_field_comparison(pod_outputs_shaped[0], pod_labels_shaped[0], save=True, fname=f"{args.encoder}_{args.decoder}_{args.dataset}_e{args.encoder_depth}_d{args.decoder_depth}_lr{args.lr:0.2e}_pod_comparison")
                plot_field_comparison(pod_outputs_shaped[0], full_labels_shaped[0], save=True, fname=f"{args.encoder}_{args.decoder}_{args.dataset}_e{args.encoder_depth}_d{args.decoder_depth}_lr{args.lr:0.2e}_full_comparison")

            # Calculate loss
            test_loss_pod += loss_fn(pod_outputs, pod_labels).item()
            if args.eval_full:
                test_loss_pod_full += loss_fn(pod_outputs_shaped, pod_labels_shaped).item()
                test_loss_full += loss_fn(pod_outputs_shaped, full_labels_shaped).item()

        # Average loss
        test_loss_pod /= len(test_dl)
        if args.eval_full:
            test_loss_pod_full /= len(test_dl)
            test_loss_full /= len(test_dl)

    if args.eval_full:
        return test_loss_pod, test_loss_pod_full, test_loss_full
    else:
        return test_loss_pod

def evaluate_model(model, test_dl, sensors, args):
    """
    Evaluate a PyTorch model.
    """
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_dl:
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            outputs = model(input_sensors)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows, w=args.data_cols, d=args.d_data)

            # Calculate loss
            test_loss += loss_fn(outputs, labels).item()

        # Average loss
        test_loss /= len(test_dl)

    return test_loss


def train_model(model, train_dl, val_dl, sensors, start_epoch, best_val, best_epoch, train_losses, val_losses, optimizer, args):
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
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        args (argparse.Namespace): Arguments to use for training.
    """
    # Set up model, optimizer, and loss
    loss_fn = torch.nn.MSELoss()
    model.to(args.device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Calculate training loss
        train_loss = 0.0
        for i, batch in enumerate(train_dl):
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            optimizer.zero_grad()
            outputs = model(input_sensors)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows, w=args.data_cols, d=args.d_data)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # Average loss
        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Calculate validation loss
        val_loss = evaluate_model(model, val_dl, sensors, args)
        val_losses.append(val_loss)

        # Save model to checkpoint if validation loss is lower than best validation loss
        if val_loss < best_val:
            if args.verbose:
                print()
                print(f'Saving model to {args.best_checkpoint_path}, validation loss improved from {best_val:0.4e} to {val_loss:0.4e}, ')
            best_val = val_loss
            best_epoch = epoch+1
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.best_checkpoint_path)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.latest_checkpoint_path)
            if args.verbose:
                print()
        
        # Save model to checkpoint if save_every_n_epochs is reached
        if (epoch + 1) % args.save_every_n_epochs == 0:
            if args.verbose:
                print()
                print(f'Saving model to {args.latest_checkpoint_path}')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, args.latest_checkpoint_path)
            if args.verbose:
                print()

        # Print loss
        if args.verbose:
            print(f'Epoch {epoch+1}, Training loss: {train_loss:0.4e}, Validation loss: {val_loss:0.4e} (best: {best_val:0.4e})')
        
        # Make plot
        plot_losses(train_losses, val_losses, best_epoch, save=True, fname=f"{args.encoder}_{args.decoder}_{args.dataset}_e{args.encoder_depth}_d{args.decoder_depth}_lr{args.lr:0.2e}_losses")

    if args.verbose:
        print(f"Training complete, best validation loss: {best_val:0.4e}")
        print()

def calculate_library_dim(latent_dim, poly_order, include_sine):
    dim = 1 # Constant term
    # Polynomial terms (using combinations with replacement)
    current_dim = latent_dim
    dim += current_dim
    if poly_order > 1:
        current_dim = current_dim * (latent_dim + 1) // 2
        dim += current_dim
    if poly_order > 2:
        current_dim = current_dim * (latent_dim + 2) // 3
        dim += current_dim
    if poly_order > 3:
        current_dim = current_dim * (latent_dim + 3) // 4
        dim += current_dim
    if poly_order > 4:
        current_dim = current_dim * (latent_dim + 4) // 5
        dim += current_dim

    if include_sine:
        dim += latent_dim
    return dim

def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    device = z.device # Get device from input tensor
    library = [torch.ones(z.shape[0], device=device)]

    # Add polynomials up to poly_order
    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i] * z[:,j]) # Use element-wise multiplication

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i] * z[:,j] * z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

    # Add sine terms if requested
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)

def min_max_scale(tensor, feature_range=(0, 1)):
    """
    Scale a tensor to a given feature range using min-max normalization.
    
    Args:
        tensor (torch.Tensor): Input tensor to be scaled
        feature_range (tuple): Desired range of transformed data (default: (0, 1))
        
    Returns:
        torch.Tensor: Scaled tensor
        tuple: (min, max) values used for scaling (for inverse transformation)
    """
    # Ensure the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # Calculate min and max
    t_min = tensor.min()
    t_max = tensor.max()
    
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
    im_shape = train[0]["input_fields"].shape
    n_steps, im_rows, im_cols, im_dim = im_shape[0], im_shape[1], im_shape[2], im_shape[3]
    
    track_count = 0

    mats = []
    for i in range(len(train)):
        data = einops.rearrange(train[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
        mats.append(data)
        track_count += 1
        if track_count >= total_tracks:
            break
        if debug:
            break
    if track_count < total_tracks:
        for i in range(len(valid)):
            data = einops.rearrange(valid[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
            mats.append(data)
            track_count += 1
            if track_count >= total_tracks:
                break
            if debug:
                break
    if track_count < total_tracks:
        for i in range(len(test)):
            data = einops.rearrange(test[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
            mats.append(data)
            track_count += 1
            if track_count >= total_tracks:
                break
            if debug:
                break
    mats = torch.cat(mats, dim=0)
    return mats

def create_mats(the_well_data, combine_all=False, debug=False):
    im_shape = the_well_data[0]["input_fields"].shape
    n_steps, im_rows, im_cols, im_dim = im_shape[0], im_shape[1], im_shape[2], im_shape[3]

    mats = []
    for i in range(len(the_well_data)):
        data = einops.rearrange(the_well_data[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
        mats.append(data)
        if debug:
            break
    if combine_all:
        mats = [torch.cat(mats, dim=0)]
    return mats

def generate_SVD(mat, n_rank=50, n_iters=2):
    U, S, V = torch.svd_lowrank(mat, n_rank, n_iters)
    return U, S, V

def create_pod(mat, V):
    pod = mat @ V
    return pod

def scale_pod(pod):
    pod_scaled, scalers = min_max_scale(pod)
    return pod_scaled, scalers

def inverse_pods_torch(pods_scaled, scalers, V, device=None):
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
    mat_hats = []
    for pod_scaled in pods_scaled:
        mat_hat = inverse_min_max_scale(pod_scaled, scalers)
        mat_hat = mat_hat @ V.T
        mat_hats.append(mat_hat)
    return mat_hats

def inverse_pod(pod_scaled, scalers, V):
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