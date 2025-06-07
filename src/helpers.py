import os
import copy
import torch
import einops
import random
import pickle
from torch import nn
from pathlib import Path
from src.plots import plot_losses, plot_field_comparison

def get_dataset_dims(dataset):
    if dataset == "sst":
        return (180, 360, 1)
    elif dataset == "planetswe":
        return(256, 512, 3)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

def print_model_size(model, name):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb_before = int(size_all_mb)
    size_all_mb_after = int((size_all_mb - size_all_mb_before) * 100)
    print(f'{name} size: {size_all_mb_before}.{size_all_mb_after:02d}MB')

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

def evaluate_model(model, dl, sensors, scalers, epoch=0, args=None, use_sindy_loss=False):
    """
    Evaluate a PyTorch model.
    """
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    model.eval()
    dl_loss = 0.0
    sindy_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            if args.dataset in ["planetswe", "gray_scott_reaction_diffusion"]:
                inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'b w n d -> b w (n d)')

            # Pass data through model
            output = model(input_sensors)

            outputs = output["output"]
            sindy_loss_batch = output.get("sindy_loss", None)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows_out, w=args.data_cols_out, d=args.d_data_out)

            # Calculate loss
            reconstruction_loss = loss_fn(outputs, labels)

            if use_sindy_loss and sindy_loss_batch is not None:
                sindy_loss_batch = args.sindy_loss_weight * sindy_loss_batch
                loss_batch = reconstruction_loss + sindy_loss_batch

            else:
                loss_batch = reconstruction_loss

            dl_loss += loss_batch.item()

            if use_sindy_loss and sindy_loss_batch is not None:
                sindy_loss += sindy_loss_batch.item()

            # Plot
            if args.generate_training_plots and i == 0:
                outputs = outputs.detach()[0]
                labels = labels[0]

                for j in range(outputs.shape[2]):
                    outputs[:,:,j] = inverse_min_max_scale(outputs[:,:,j], scalers[j])
                    labels[:,:,j] = inverse_min_max_scale(labels[:,:,j], scalers[j])

                if use_sindy_loss:
                    ident = "val"
                else:
                    ident = "test"

                plot_field_comparison(outputs, labels, dataset=args.dataset, sensors=sensors, save=True, fname=f"{args.identifier}_full_comparison_epoch{epoch}_{ident}")

        # Average loss
        dl_loss /= len(dl)
        if use_sindy_loss and sindy_loss_batch is not None:
            sindy_loss /= len(dl)

    return dl_loss, sindy_loss

def create_plots(model, ds, sensors, metadata, args=None):
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
            inputs, labels = ds[i]["input_fields"], ds[i]["output_fields"][0,:,:,:]
            if args.dataset in ["planetswe", "gray_scott_reaction_diffusion"]:
                inputs, labels = inputs.to(args.device), labels.to(args.device)

            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)
    
            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'w n d -> 1 w (n d)')

            # Pass data through model
            output = model(input_sensors)

            outputs = output["output"]

            # Reshape output
            outputs = einops.rearrange(outputs, '1 (r w d) -> r w d', r=args.data_rows_out, w=args.data_cols_out, d=args.d_data_out)

            # Convert back to original scale (except for plasma)
            if args.dataset not in ['plasma']:
                for j in range(outputs.shape[2]):
                    outputs[:,:,j] = inverse_min_max_scale(outputs[:,:,j], metadata['scalers'][j])
                    labels[:,:,j] = inverse_min_max_scale(labels[:,:,j], metadata['scalers'][j])

                plot_field_comparison(outputs, labels, dataset=args.dataset, sensors=sensors, save=True, fname=f"{args.identifier}_full_comparison_{i}")
            elif args.dataset in ['plasma']:
                # For each feature ...
                for k in range(14):
                    # Convert from V to full space
                    u = torch.from_numpy(metadata['u_total'][20*k:20*(k+1),:]).float().to(args.device)
                    s = torch.from_numpy(metadata['s_total'][:,k]).float().to(args.device)
                    v = torch.from_numpy(metadata['v_total'][:,20*k:20*(k+1)]).float().to(args.device)

                    true_shaped = (labels[0,20*k:20*(k+1),0] @ torch.diag(s) @ u)
                    output_shaped = (outputs[0,20*k:20*(k+1),0] @ torch.diag(s) @ u)

                    true_shaped = einops.rearrange(true_shaped, '(r c) -> c r ()', r = args.data_rows_in, c = args.data_cols_in)
                    output_shaped = einops.rearrange(output_shaped, '(r c) -> c r ()', r = args.data_rows_in, c = args.data_cols_in)

                    plot_field_comparison(output_shaped, true_shaped, dataset=args.dataset, sensors=sensors, save=True, fname=f"{args.identifier}_f{k+1}_full_comparison_{i}")

def train_model(model, train_dl, val_dl, sensors, start_epoch, best_val, best_epoch, train_losses, val_losses, optimizer, scalers, args):
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
        scalers (list): List of tuples of (min, max) values used for scaling (for inverse transformation) for each dimension
        args (argparse.Namespace): Arguments to use for training.
    """
    # Set up model, optimizer, and loss
    loss_fn = torch.nn.MSELoss()
    model.to(args.device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Calculate training loss
        train_loss = 0.0
        sindy_loss = 0.0

        for i, batch in enumerate(train_dl):
            # Get raw data
            inputs, labels = batch["input_fields"], batch["output_fields"][:,0,:,:,:]
            if args.dataset in ["planetswe", "gray_scott_reaction_diffusion"]:
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
            output = model(input_sensors)

            outputs = output["output"]
            sindy_loss_batch = output.get("sindy_loss", None)

            # Reshape output
            outputs = einops.rearrange(outputs, 'b (r w d) -> b r w d', b=inputs.shape[0], r=args.data_rows_out, w=args.data_cols_out, d=args.d_data_out)

            # Calculate loss
            reconstruction_loss = loss_fn(outputs, labels)

            # Add other losses if available
            loss = reconstruction_loss
            if sindy_loss_batch is not None:
                sindy_loss_batch = args.sindy_loss_weight * sindy_loss_batch
                loss += sindy_loss_batch
            if args.encoder in ["sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]:
                if args.sindy_attention_weight > 0.0:
                    sindy_sum = args.sindy_attention_weight * model.encoder.get_SINDy_coefficients_sum()
                    loss += sindy_sum

            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if sindy_loss_batch is not None:
                sindy_loss += sindy_loss_batch.item()

            # Plot
            if args.generate_training_plots and i == 0:
                outputs = outputs.detach()[0]
                labels = labels[0]

                for j in range(outputs.shape[2]):
                    outputs[:,:,j] = inverse_min_max_scale(outputs[:,:,j], scalers[j])
                    labels[:,:,j] = inverse_min_max_scale(labels[:,:,j], scalers[j])

                plot_field_comparison(outputs, labels, dataset=args.dataset, sensors=sensors, save=True, fname=f"{args.identifier}_full_comparison_epoch{epoch}")

        # Threshold if necessary
        if args.encoder in ["sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]:
            if epoch > 0 and (epoch+1) % args.sindy_attention_threshold_epoch == 0:
                print(f"Thresholding SINDy coefficients (epoch {epoch+1})")
                model.encoder.threshold_all_layers(args.sindy_attention_threshold)

        # Average loss
        train_loss /= len(train_dl)
        if sindy_loss_batch is not None:
            sindy_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Calculate validation loss
        val_loss, sindy_val_loss = evaluate_model(model, val_dl, sensors, epoch=epoch, scalers=scalers, args=args, use_sindy_loss=True)
        val_losses.append(val_loss)

        # Save model to checkpoint if validation loss is lower than best validation loss
        if epoch > args.early_stop and val_loss < best_val:
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
                'sensors': sensors,
            }, args.best_checkpoint_path)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'sensors': sensors,
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
                'sensors': sensors
            }, args.latest_checkpoint_path)
            if args.verbose:
                print()

        # Print loss
        if args.verbose:
            print(f'Epoch {epoch+1}, Training loss: {train_loss:0.4e}, Validation loss: {val_loss:0.4e} (best: {best_val:0.4e})')
            if sindy_loss_batch is not None:
                print(f'Epoch {epoch+1}, SINDy training loss: {sindy_loss:0.4e}, SINDy validation loss: {sindy_val_loss:0.4e}')

        # Print model coefficients
        if args.verbose and args.encoder == "sindy_attention_transformer":
            print_model_coefficients(model, args)

        # Make plot
        plot_losses(train_losses, val_losses, best_epoch, save=True, fname=f"{args.identifier}_losses")

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

def sindy_library_terms(latent_dim, poly_order, include_sine=False):
    # Ones - constant term
    library = ["1"]

    # Add polynomials up to poly_order
    for i in range(latent_dim):
        library.append(f"z{i}")

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(f"z{i} * z{j}") # Use element-wise multiplication

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(f"z{i} * z{j} * z{k}")

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(f"z{i} * z{j} * z{k} * z{p}")

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(f"z{i} * z{j} * z{k} * z{p} * z{q}")

    # Add sine terms if requested
    if include_sine:
        for i in range(latent_dim):
            library.append(f"sin(z{i})")

    return library

def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    device = z.device # Get device from input tensor

    # Ones - constant term
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
    im_shape = train[0]["input_fields"].shape
    n_steps, im_rows, im_cols, im_dim = im_shape[0], im_shape[1], im_shape[2], im_shape[3]
    
    track_count = 0

    mats = []
    for i in range(len(train)):
        #data = einops.rearrange(train[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
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
            #data = einops.rearrange(test[i]["input_fields"], "t r c d -> t (r c d)", t=n_steps, r=im_rows, c=im_cols, d=im_dim)
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
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            results.append(data)
    return results

def get_result_loss(result):
    return result.get('test_loss', result.get('test_loss_pod', None))

def print_model_coefficients(model, args):
    library = model.encoder.encoder.layers[0].self_attn.library_terms
    for i in range(args.encoder_depth):
        print(f"Layer {i}:")
        for j in range(args.n_heads):
            print(f"Head {j}:")
            for k in range(args.hidden_size // args.n_heads):
                print(f"Hidden layer {k}:")
                output_str = ""
                for l in range(len(library)):
                    output_str += f"{model.encoder.encoder.layers[i].self_attn.coefficients[j][l][k].item():0.3f} \\cdot {library[l]} + "
                print(output_str[:-3])
            print()

def get_top_N_models_by_loss(dataset_name, pickle_dir, loss_only=False, N=5):
    """
    Returns the top 5 models (filenames) with the lowest test loss for a given dataset.
    
    Args:
        dataset_name (str): The dataset name to filter by (e.g., 'sst').
        loss_only (bool): If True, only return the loss value.
        pickles_dir (str): Path to the pickles directory.
        N (int): Number of results to return.
        
    Returns:
        List of tuples: [(filename, loss), ...] sorted by lowest loss.
    """
    results = []
    for fname in os.listdir(pickle_dir):
        if dataset_name in fname and fname.endswith('.pkl'):
            fpath = os.path.join(pickle_dir, fname)
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
                # Try both 'test_loss' and 'test_loss_pod' keys for robustness
                results.append((fname, data))
    # Sort by loss (ascending) and return top 5
    results.sort(key=lambda x: get_result_loss(x[1]), reverse=False)
    if loss_only:
        results = [(x[0], get_result_loss(x[1])) for x in results]
    return results[:N]

def get_identifier(filename):
    """Extract identifier from filename by removing extension and _test_loss suffix."""
    name = Path(filename).stem  # Remove extension
    if name.endswith('_test_loss'):
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
    t = torch.linspace(0, 2*torch.pi, T)
    
    # Initialize output tensor
    output = torch.zeros((X, T))
    
    # Generate each time series
    for i in range(X):
        # Sum all sinusoids
        for j in range(n_sin):
            # Generate random frequencies and amplitudes
            frequencies = torch.rand(n_sin) * 4 * torch.pi  # Random frequencies between 0 and 2π
            amplitudes = torch.rand(n_sin) * 2  # Random amplitudes between 0 and 2
            output[i] += amplitudes[j] * torch.sin(frequencies[j] * t)
    
    return output

# We use this for exact parity with the PyTorch implementation, having the same init
# for every layer might not be necessary.
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])