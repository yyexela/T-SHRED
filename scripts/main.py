###########
# Imports #
###########

import sys
import time
import torch
import pickle
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
    args.d_data_in = train_ds[0]['input_fields'].shape[-1]
    args.data_rows_in, args.data_cols_in = (train_ds[0]['input_fields'].shape[-3],
                                      train_ds[0]['input_fields'].shape[-2])
    args.d_data_out = train_ds[0]['output_fields'].shape[-1]
    args.data_rows_out, args.data_cols_out = (train_ds[0]['output_fields'].shape[-3],
                                      train_ds[0]['output_fields'].shape[-2])
    args.d_model = args.n_sensors * args.d_data_in
    args.dim_feedforward = args.hidden_size * 4
    args.output_size = args.data_rows_out*args.data_cols_out*args.d_data_out

    # Create dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Save model location
    latest_model_name = f'{args.identifier}_latest.pt'
    best_model_name = f'{args.identifier}_best.pt'
    args.latest_checkpoint_path = checkpoint_dir / latest_model_name
    args.best_checkpoint_path = checkpoint_dir / best_model_name

    # Load model if checkpoint exists
    model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses, sensors = models.load_model_from_checkpoint(args.latest_checkpoint_path, args=args)

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
        optimizer=optimizer,
        scalers=metadata['scalers'],
        args=args
    )

    # Clean up variables after training
    del train_ds, val_ds, train_dl, val_dl, model, optimizer
    torch.cuda.empty_cache()
    time.sleep(1.0)

    # Evaluate best validation model
    best_model, _, start_epoch, best_val, best_epoch, train_losses, val_losses, sensors = models.load_model_from_checkpoint(args.best_checkpoint_path, force_load=True, args=args)

    # Threshold
    if args.encoder in ["sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]:
        print(f"Thresholding SINDy coefficients")
        best_model.encoder.threshold_all_layers(args.sindy_attention_threshold)

    # Print model coefficients
    if args.verbose and args.encoder == "sindy_attention_transformer":
        helpers.print_model_coefficients(best_model, args)

    # Calculate loss
    test_loss, _ = helpers.evaluate_model(best_model, test_dl, sensors, metadata['scalers'], args=args, use_sindy_loss=False)
    if args.verbose:
        print(f'Test loss: {test_loss:0.4e}')
    save_dict = {'test_loss': test_loss, 'start_epoch': start_epoch, 'best_val': best_val, 'best_epoch': best_epoch, 'train_losses': train_losses, 'val_losses': val_losses, 'sensors': sensors}

    # Create plots
    if args.generate_test_plots:
        helpers.create_plots(best_model, test_ds, sensors, metadata, args=args)

    # Save pickle
    with open(pickle_dir / f'{args.identifier}.pkl', 'wb') as f:
        save_dict['hyperparameters'] = vars(args)
        pickle.dump(save_dict, f)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help="Dataset batch size")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run (active_matter, active_matter_pod, planetswe, planetswe_pod, sst, sst_demo, plasma)")
    parser.add_argument('--decoder', type=str, default="mlp", help="Which decoder to use (unet, mlp)")
    parser.add_argument('--decoder_depth', type=int, default=2, help="Number of decoder layers")
    parser.add_argument('--device', type=str, default="cuda:2", help="Which device to run on")
    parser.add_argument('--dropout', type=float, default=0.1, help="Model droput proportion")
    parser.add_argument('--dt', type=float, default=1.0, help="Time step for SINDy derivatives (Euler integration)")
    parser.add_argument('--early_stop', type=int, default=0, help="Train the model for at least this many epochs before saving best validation score")
    parser.add_argument('--encoder', type=str, default="transformer", help="Which encoder to use (lstm, gru, transformer, sindy_attention_transformer, sindy_loss_transformer)")
    parser.add_argument('--eval_full', action='store_true', help="Evaluate on full dataset (BAD FOR RAM)")
    parser.add_argument('--encoder_depth', type=int, default=3, help="Number of encoder layers")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--hidden_size', type=int, default=12, help="Hidden size of encoder")
    parser.add_argument('--generate_test_plots', action='store_true', help="Generate test plots")
    parser.add_argument('--generate_training_plots', action='store_true', help="Generate training plots")
    parser.add_argument('--include_sine', action='store_true', help="Include sine in transformer SINDy library")
    parser.add_argument('--identifier', type=str, required=True, help="Identifier for logging")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument('--n_heads', type=int, default=6, help="Number of transformer heads")
    parser.add_argument('--n_sensors', type=int, default=50, help="Number of sensors")
    parser.add_argument('--n_well_tracks', type=int, default=10, help="Maximum number of tracks to load from the well dataset")
    parser.add_argument('--poly_order', type=int, default=2, help="Order of polynomial library for SINDy transformer library")
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help="After how many epochs to checkpoint model")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--sindy_attention_threshold', type=float, default=0.05, help="Threshold for SINDy coefficient sparsification (attention)")
    parser.add_argument('--sindy_attention_threshold_epoch', type=int, default=10, help="Every n epochs to threshold SINDy coefficients (attention)")
    parser.add_argument('--sindy_attention_weight', type=float, default=100, help="Weight for SINDy attention term")
    parser.add_argument('--sindy_loss_threshold', type=float, default=0.05, help="Threshold for SINDy coefficient sparsification (loss)")
    parser.add_argument('--sindy_loss_weight', type=float, default=100, help="Weight for SINDy loss term")
    parser.add_argument('--skip_load_checkpoint', action='store_true', help="Skip loading checkpoint")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose messages")
    parser.add_argument('--window_length', type=int, default=10, help="Dataset window length")
    args = parser.parse_args()

    main(args)
        
