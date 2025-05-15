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

#UTransformer = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)

def main(args=None):
    # Load pickle
    fpath = os.path.join(pickle_dir, args.pickle_file)
    with open(fpath, 'rb') as f:
        hp = pickle.load(f)["hyperparameters"]
        hp = Namespace(**hp)
        print(hp)

    # Load dataset
    if 'pod' in hp.dataset:
        # POD more complicated due to V and scaler to bring back to original space
        # first three datasets are POD, last three are not (they are full)
        if hp.eval_full:
            train_ds, val_ds, test_ds, train_full_ds, valid_full_ds, test_full_ds, (V, scaler, im_dims) = datasets.load_dataset(hp)
        else:
            train_ds, val_ds, test_ds, (V, scaler, im_dims) = datasets.load_dataset(hp)
    else:
        train_ds, val_ds, test_ds, (mean, std) = datasets.load_dataset(hp)

    # Generate sensors
    sensors = helpers.generate_sensor_positions(hp.n_sensors, hp.data_rows, hp.data_cols)

    # Create dataloader
    test_dl = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)
    test_full_dl = DataLoader(test_full_ds, batch_size=hp.batch_size, shuffle=False)

    # Load model
    model, optimizer, start_epoch, best_val, best_epoch, train_losses, val_losses = models.load_model_from_checkpoint(hp.best_checkpoint_path, hp)

    model.eval()

    # Generate plot for POD
    if 'pod' in hp.dataset:
        with torch.no_grad():
            # Set up iterators for dual dataset loading
            full_iterator = iter(test_full_dl)
            pod_iterator = iter(test_dl)
            num_iters = len(test_dl)

            # Get batch
            full_batch = next(full_iterator)
            pod_batch = next(pod_iterator)

            # Get data
            pod_inputs, pod_labels = pod_batch["input_fields"], pod_batch["output_fields"][:,0,:,:,:]
            pod_inputs, pod_labels = pod_inputs.to(hp.device), pod_labels.to(hp.device)

            full_inputs, full_labels = full_batch["input_fields"], full_batch["output_fields"][:,0,:]
            full_inputs, full_labels = full_inputs.to(hp.device), full_labels.to(hp.device)

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
            pod_outputs = einops.rearrange(pod_outputs, 'b (r w d) -> b r w d', b=batch_size, r=hp.data_rows, w=hp.data_cols, d=hp.d_data)
            
            # Remove singular dimensions
            pod_outputs_squeezed = pod_outputs[:,0,:,0]
            pod_labels_squeezed = pod_labels[:,0,:,0]

            # Inverse POD to get full scale image
            pod_outputs_full = helpers.inverse_pods_torch(pod_outputs_squeezed, scaler, V, device=hp.device)
            pod_labels_full = helpers.inverse_pods_torch(pod_labels_squeezed, scaler, V, device=hp.device)

            # Convert back to original shape
            pod_outputs_shaped = einops.rearrange(pod_outputs_full, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])
            pod_labels_shaped = einops.rearrange(pod_labels_full, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])
            full_labels_shaped = einops.rearrange(full_labels, "b (r c d) -> b r c d", b=batch_size, r=im_dims[0], c=im_dims[1], d=im_dims[2])

            # Generate plots
            plots.plot_field_comparison(pod_outputs_shaped[0], pod_labels_shaped[0], save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_pod_comparison")
            plots.plot_field_comparison(pod_outputs_shaped[0], full_labels_shaped[0], save=True, fname=f"{hp.encoder}_{hp.decoder}_{hp.dataset}_e{hp.encoder_depth}_d{hp.decoder_depth}_lr{hp.lr:0.2e}_full_comparison")
    else:
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file', type=str, default=None, help="Pickle file from associated run")
    args = parser.parse_args()
    main(args)       
        
