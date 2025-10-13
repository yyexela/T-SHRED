###########
# Imports #
###########

import sys
import time
import yaml
import torch
import pickle
import einops
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Bug workaround, see https://github.com/pytorch/pytorch/issues/16831
torch.backends.cudnn.benchmark = False

# Local files
pkg_path = str(Path(__file__).parent.parent)
sys.path.insert(0, pkg_path)

from src import *

###############
# Directories #
###############

top_dir = Path(__file__).parent.parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'
checkpoint_dir = top_dir / 'checkpoints'
pickle_dir = top_dir / 'pickles'

fig_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
pickle_dir.mkdir(parents=True, exist_ok=True)

########
# Main #
########

def main(args=None):
    # Verify args
    helpers.verify_args(args)

    # Set Seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set CUDA seeds if using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Make CuDNN deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    train_ds, val_ds, test_ds, metadata = datasets.load_dataset(args)
    args.d_data_in = train_ds[0][0].shape[-1]
    args.data_rows_in, args.data_cols_in = (train_ds[0][0].shape[-3],
                                      train_ds[0][0].shape[-2])
    args.d_data_out = train_ds[0][1].shape[-1]
    args.data_rows_out, args.data_cols_out = (train_ds[0][1].shape[-3],
                                      train_ds[0][1].shape[-2])
    args.d_model = args.n_sensors * args.d_data_in
    args.dim_feedforward = args.hidden_size * 2
    args.output_size = args.data_rows_out*args.data_cols_out*args.d_data_out

    # Create dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Save model location
    latest_model_name = f'{args.identifier}_latest.pt'
    best_model_name = f'{args.identifier}_best.pt'
    args.latest_checkpoint_path = checkpoint_dir / latest_model_name
    args.best_checkpoint_path = checkpoint_dir / best_model_name

    # Load model if checkpoint exists
    model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses, model_eigvs, sensors = models.load_model_from_checkpoint(args.latest_checkpoint_path, args=args)

    # Print hyperparameters
    helpers.print_dictionary(vars(args), 'Hyperparameters:')

    # Print model size
    helpers.print_model_size(model, "Full")
    helpers.print_model_size(model.encoder, "Encoder")
    helpers.print_model_size(model.decoder, "Decoder")
    print()

    # Train model
    helpers.train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        sensors=sensors,
        start_epoch=start_epoch,
        best_val=best_val,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        model_eigvs=model_eigvs,
        optimizer=optimizer,
        metadata=metadata,
        args=args
    )

    # Clean up variables after training
    del train_ds, val_ds, train_dl, val_dl, model, optimizer
    torch.cuda.empty_cache()
    time.sleep(1.0)

    # Evaluate best validation model
    best_model, _, start_epoch, best_val, best_epoch, train_losses, val_losses, model_eigvs, sensors = models.load_model_from_checkpoint(args.best_checkpoint_path, force_load=True, args=args)

    # Threshold
    if args.encoder in ["sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]:
        if args.verbose:
            print(f"Thresholding SINDy coefficients")
        helpers.threshold_all_layers(best_model.encoder, args.sindy_attention_threshold, verbose=args.verbose)

    # Print model coefficients
    if args.verbose and (args.encoder in ["sindy_attention_transformer"]):
        helpers.print_model_coefficients(best_model, args)

    # Calculate loss
    test_loss_next, _ = helpers.evaluate_model(best_model, test_dl, sensors, metadata, rollout=False, rmse=True, args=args)
    print(f'Test loss (next rmse): {test_loss_next:0.4e}')

    test_loss_rollout, _ = helpers.evaluate_model(best_model, test_dl, sensors, metadata, rollout=True, rmse=False, args=args)
    print(f'Test loss (rollout rmse): {test_loss_rollout:0.4e}')

    save_dict = {'test_loss_next': test_loss_next, 'test_loss_rollout': test_loss_rollout, 'start_epoch': start_epoch, 'best_val': best_val, 'best_epoch': best_epoch, 'train_losses': train_losses, 'val_losses': val_losses, 'model_eigvs': model_eigvs, 'sensors': sensors}

    # Create plots
    if args.generate_test_plots:
        #helpers.create_next_step_plots(best_model, test_ds, sensors, metadata, args=args)
        if "sindy_attention" in args.encoder:
            model_eigvs = np.asarray(model_eigvs)
            model_eigvs = einops.rearrange(model_eigvs, 'epochs heads coeffs -> heads epochs coeffs')
            for i in range(args.n_heads):
                plots.plot_eigvs(model_eigvs[i], save=True, fname=f"{args.identifier}_eigvs_head{i}")
        helpers.create_far_out_plots(best_model, test_ds, sensors, metadata, args=args)

    # Save pickle
    with open(pickle_dir / f'{args.identifier}.pkl', 'wb') as f:
        save_dict['hyperparameters'] = vars(args)
        pickle.dump(save_dict, f)

    # Delete checkpoint after training
    if args.delete_checkpoint:
        args.latest_checkpoint_path.unlink(missing_ok=True)
        args.best_checkpoint_path.unlink(missing_ok=True)

def config_main(config_str: str):
    """
    Helper for hyperparameter tuning, just takes in path to config file
    """
    with open(config_str, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    args = argparse.Namespace(**model_config)
    args.config = config_str
    main(args)

if __name__ == '__main__':
    args = helpers.parse_args()
    main(args)
        
