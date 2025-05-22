###########
# Imports #
###########

import os
import sys
import torch
import pickle
import random
import einops
import argparse
from pathlib import Path
from argparse import Namespace

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
pickle_dir = top_dir / 'pickles'

fig_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
pickle_dir.mkdir(parents=True, exist_ok=True)

########
# Main #
########

def main(args=None):
    # Set Seed
    torch.manual_seed(0)
    random.seed(0)

    # Load pickle
    fpath = os.path.join(pickle_dir, f"{args.identifier}.pkl")
    with open(fpath, 'rb') as f:
        result = pickle.load(f)
        hp = Namespace(**result['hyperparameters'])
        print(hp)

    # Switch device
    hp.device = 'cuda:2'
    im_dims = helpers.get_dataset_dims(hp.dataset)

    # Set up dataset and iterator
    if hp.dataset in ["planetswe_full"]:
        train_ds, val_ds, test_ds, (V, scaler, im_dims) = datasets.load_dataset(hp)
    else:
        train_ds, val_ds, test_ds, (scaler) = datasets.load_dataset(hp)
    if args.split == 'test':
        ds = test_ds
    elif args.split == 'val':
        ds = val_ds
    else:
        ds = train_ds
    if hp.dataset == "plasma":
        ds_iter = range(len(ds))
    else:
        ds_iter = [0, len(ds)-1]

    # Get sensors
    sensors = result['sensors']

    # Load model
    latest_model_name = f'{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_p{hp.poly_order}_model_latest.pt'
    best_model_name = f'{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_p{hp.poly_order}_model_best.pt'
    hp.latest_checkpoint_path = checkpoint_dir / latest_model_name
    hp.best_checkpoint_path = checkpoint_dir / best_model_name
    print("Checking ", hp.best_checkpoint_path)
    model, _, _, _, _, _, _ = models.load_model_from_checkpoint(hp.best_checkpoint_path, hp)

    model.eval()

    # Collect plot data
    if hp.dataset == "plasma":
        # Collect all outputs from test trajectory
        expected_trajectory = list()
        output_trajectory = list()
    with torch.no_grad():
        for i in ds_iter:
            # Get raw data
            inputs, labels = ds[i]["input_fields"], ds[i]["output_fields"][0,:,:,:]
            if hp.dataset in ["planetswe", "gray_scott_reaction_diffusion"]:
                inputs, labels = inputs.to(hp.device), labels.to(hp.device)

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
            outputs = einops.rearrange(outputs, '1 (r w d) -> r w d', r=hp.data_rows, w=hp.data_cols, d=hp.d_data)

            if hp.dataset == "plasma":
                # Collect
                expected_trajectory.append(labels)
                output_trajectory.append(outputs)
            elif hp.dataset in ["sst", "planetswe_full"]:
                # Convert back to original scale
                outputs = helpers.inverse_min_max_scale(outputs, scaler)
                labels = helpers.inverse_min_max_scale(labels, scaler)

                if hp.dataset == "planetswe_full":
                    outputs = einops.rearrange(outputs, "1 (r w d) 1 -> r w d", r=im_dims[0], w=im_dims[1], d=im_dims[2])
                    labels = einops.rearrange(labels, "1 (r w d) 1 -> r w d", r=im_dims[0], w=im_dims[1], d=im_dims[2])

                plots.plot_field_comparison(outputs, labels, dataset=hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_full_comparison_{i}")

    if hp.dataset == "plasma":
        # Only plasma here
        expected_trajectory = torch.cat(expected_trajectory, dim=0)
        output_trajectory = torch.cat(output_trajectory, dim=0)
        output_trajectory = einops.rearrange(output_trajectory, "n d -> n d 1")

        # Make plot
        plots.plot_field_comparison(output_trajectory, expected_trajectory, hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_comparison")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, help="Identifier of the associated run")
    parser.add_argument('--split', type=str, default='test', help="Split to use for plotting")
    args = parser.parse_args()
    main(args)       
        
