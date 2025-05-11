###########
# Imports #
###########

import os
import sys
import torch
import argparse
import datetime
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from torch.distributed import destroy_process_group
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data.distributed import DistributedSampler

# Bug workaround, see https://github.com/pytorch/pytorch/issues/16831
torch.backends.cudnn.benchmark = False

# Local files
pkg_path = str(Path(__file__).parent.parent)
sys.path.insert(0, pkg_path)

from src import *

###############
# Directories #
###############

# Directories
top_dir = Path(__file__).parent.parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'
checkpoint_dir = top_dir / 'checkpoints'

checkpoint_dir.mkdir(parents=True, exist_ok=True)

########
# Main #
########

def main(args=None):
    # Set up DDP
    helpers.ddp_setup()

    # Load dataset
    train_ds, val_ds, test_ds, (mean, std) = datasets.load_dataset(args)
    args.gpu_id = int(os.environ["LOCAL_RANK"])
    args.d_data = train_ds[0]['input_fields'].shape[-1]
    args.data_rows, args.data_cols = (train_ds[0]['input_fields'].shape[-3],
                                      train_ds[0]['input_fields'].shape[-2])
    args.d_model = args.n_sensors * args.d_data
    args.dim_feedforward = args.hidden_size * 4
    args.output_size = args.data_rows*args.data_cols*args.d_data

    if args.gpu_id == 0:
        args.verbose = args.verbose
    else:
        args.verbose = False

    # Generate sensors
    sensors = helpers.generate_sensor_positions(args.n_sensors, args.data_rows, args.data_cols)

    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(train_ds),
        num_workers=2
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(val_ds),
        num_workers=2
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_ds),
        num_workers=2
    )

    # Save model location
    #time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    latest_model_name = f'{args.encoder}_{args.decoder}_{args.dataset}_{args.lr:0.2e}_model_latest.pt'
    best_model_name = f'{args.encoder}_{args.decoder}_{args.dataset}_{args.lr:0.2e}_model_best.pt'
    args.latest_checkpoint_path = checkpoint_dir / latest_model_name
    args.best_checkpoint_path = checkpoint_dir / best_model_name

    # Load model if checkpoint exists
    model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses = models.load_model_from_checkpoint(args.latest_checkpoint_path, args)

    # Print hyperparameters
    if args.verbose:
        helpers.print_dictionary(vars(args), 'Hyperparameters:')

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
        args=args
    )

    # Evaluate best validation model
    model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses = models.load_model_from_checkpoint(args.best_checkpoint_path, args)
    test_loss = helpers.evaluate_model(model, test_dl, sensors, args)
    if args.verbose:
        print(f'Test loss: {test_loss:0.4e}')

    # Clean up DDP
    destroy_process_group()

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help="Dataset batch size")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run (active_matter, planetswe, sst, sst_demo, plasma)")
    parser.add_argument('--decoder', type=str, default="mlp", help="Which decoder to use (unet, mlp)")
    parser.add_argument('--decoder_depth', type=int, default=2, help="Number of decoder layers")
    parser.add_argument('--device', type=str, default="cuda:2", help="Which device to run on")
    parser.add_argument('--dropout', type=float, default=0.1, help="Model droput proportion")
    parser.add_argument('--encoder', type=str, default="transformer", help="Which encoder to use (lstm, transformer, sindy_attention_transformer, sindy_loss_transformer)")
    parser.add_argument('--encoder_depth', type=int, default=3, help="Number of encoder layers")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--hidden_size', type=int, default=12, help="Hidden size of encoder")
    parser.add_argument('--include_sine', action='store_true', help="Include sine in transformer SINDy library")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument('--n_heads', type=int, default=6, help="Number of transformer heads")
    parser.add_argument('--n_sensors', type=int, default=50, help="Number of sensors")
    parser.add_argument('--poly_order', type=int, default=2, help="Order of polynomial library for SINDy transformer library")
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help="After how many epochs to checkpoint model")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose messages")
    parser.add_argument('--window_length', type=int, default=10, help="Dataset window length")
    args = parser.parse_args()
    main(args)       
        
