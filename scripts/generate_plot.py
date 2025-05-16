###########
# Imports #
###########

import os
import sys
import torch
import pickle
import einops
import argparse
import pprint as pp
from pathlib import Path
from argparse import Namespace
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
    # Load pickle
    fpath = os.path.join(pickle_dir, f"{args.identifier}_test_loss.pkl")
    with open(fpath, 'rb') as f:
        result = pickle.load(f)
        hp = Namespace(**result['hyperparameters'])
        print(hp)

    # Set up dataset and iterator
    if "pod" in hp.dataset:
        hp.eval_full = True
        pod_ds, pod_full_ds, (V, scaler, im_dims) = datasets.load_dataset_track_pod(hp, track=0, split='test')
        ds_iter = [0, len(pod_ds)-1]
    else:
        train_ds, val_ds, test_ds, (scaler) = datasets.load_dataset(hp)
        if args.split == 'test':
            ds = test_ds
        elif args.split == 'val':
            ds = val_ds
        else:
            ds = train_ds
        im_dims = (hp.data_rows, hp.data_cols, hp.d_data)
        if hp.dataset == "plasma":
            ds_iter = range(len(ds))
        else:
            ds_iter = [0, len(ds)-1]

    # Get sensors
    if 'sensors' in result:
        sensors = result['sensors']
    else:
        sensors = helpers.generate_sensor_positions(hp.n_sensors, hp.data_rows, hp.data_cols)

    # Load model
    latest_model_name = f'{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_model_latest.pt'
    best_model_name = f'{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_model_best.pt'
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
            # Get data
            if "pod" in hp.dataset:
                inputs, pod_labels = pod_ds[0]["input_fields"], pod_ds[0]["output_fields"]
                inputs, pod_labels = inputs.to(hp.device), pod_labels.to(hp.device)

                full_inputs, full_labels = pod_full_ds[0]["input_fields"], pod_full_ds[0]["output_fields"]
                full_inputs, full_labels = full_inputs.to(hp.device), full_labels.to(hp.device)
            else:
                inputs, labels = ds[i]["input_fields"], ds[i]["output_fields"][0,:,:,:]
                inputs, labels = inputs.to(hp.device), labels.to(hp.device)
            
            # Extract sensors per input tensor
            input_sensors = []
            for sensor in sensors:
                input_sensors.append(inputs[:,sensor[0],sensor[1],:])
            input_sensors = torch.stack(input_sensors, dim=2)

            # Prepare input for model
            input_sensors = einops.rearrange(input_sensors, 'w n d -> 1 w (n d)')

            # Pass data through model
            outputs = model(input_sensors)

            if hp.dataset == "plasma":
                # Collect
                expected_trajectory.append(labels)
                output_trajectory.append(outputs)
            elif hp.dataset == "sst":
                # Convert back to original shape
                outputs_shaped = einops.rearrange(outputs, "1 (r c d) -> r c d", r=im_dims[0], c=im_dims[1], d=im_dims[2])

                # Convert back to original scale
                outputs_shaped = helpers.inverse_min_max_scale(outputs_shaped, scaler)
                labels = helpers.inverse_min_max_scale(labels, scaler)

                plots.plot_field_comparison(outputs_shaped, labels, dataset=hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_full_comparison_{i}")
            elif "pod" in hp.dataset:
                # Reshape output (will be: batch x 1 x dim x 1)
                pod_outputs = einops.rearrange(outputs, '1 (r w d) -> 1 r w d', r=hp.data_rows, w=hp.data_cols, d=hp.d_data)
                
                # Remove singular dimensions
                pod_outputs_squeezed = pod_outputs[:,0,:,0]
                pod_labels_squeezed = pod_labels[:,0,:,0]
                full_labels_squeezed = full_labels[:,0,:,0]

                # Inverse POD to get full scale image
                pod_outputs_full = helpers.inverse_pods_torch(pod_outputs_squeezed, scaler, V, device=hp.device)
                pod_labels_full = helpers.inverse_pods_torch(pod_labels_squeezed, scaler, V, device=hp.device)

                # Convert back to original shape
                pod_outputs_shaped = einops.rearrange(pod_outputs_full, "1 (r c d) -> r c d", r=im_dims[0], c=im_dims[1], d=im_dims[2])
                pod_labels_shaped = einops.rearrange(pod_labels_full, "1 (r c d) -> r c d", r=im_dims[0], c=im_dims[1], d=im_dims[2])
                full_labels_shaped = einops.rearrange(full_labels_squeezed, "1 (r c d) -> r c d", r=im_dims[0], c=im_dims[1], d=im_dims[2])

                # Generate plots
                plots.plot_field_comparison(pod_outputs_shaped, pod_labels_shaped, dataset=hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_pod_comparison_{i}")
                plots.plot_field_comparison(pod_outputs_shaped, full_labels_shaped, dataset=hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_full_comparison_{i}")

    if hp.dataset == "plasma":
        # Only plasma here
        expected_trajectory = torch.cat(expected_trajectory, dim=0)
        output_trajectory = torch.cat(output_trajectory, dim=0)
        output_trajectory = einops.rearrange(output_trajectory, "n d -> n d 1")

        print(expected_trajectory.shape, output_trajectory.shape)

        # Make plot
        plots.plot_field_comparison(output_trajectory, expected_trajectory, hp.dataset, save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_comparison")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, help="Identifier of the associated run")
    parser.add_argument('--split', type=str, default='test', help="Split to use for plotting")
    args = parser.parse_args()
    main(args)       
        
