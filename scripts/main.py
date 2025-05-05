###########
# Imports #
###########

import sys
import torch
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

# Local files
pkg_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, pkg_path)

from src import *

###############
# Directories #
###############

# Directories
top_dir = Path(__file__).parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

########
# Main #
########

def main(args=None):
    train_dl ,valid_dl, test_dl = datasets.load_dataset(args.dataset)
    d_model, h_head = datasets.get_dataset_hyperparameters(args.dataset)
    model = models.get_model(args.model)

    # We train the model using the training and validation datasets.

    # Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.

    UTransformer = models.TimeSeries_UTransformer(d_model=d_model, n_heads=n_heads, sequence_length=args.seq_length, dropout=args.dropout).to(args.device)

    #validation_errors = models.fit(UTransformer, train_dataset, valid_dataset, batch_size=25, num_epochs=8, lr=0.001, verbose=True, patience=5)
    #UTransformer = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(UTransformer, train_dl, valid_dl, batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
    print(list(validation_errors))

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run (MNIST, FMNIST, or YaleFaces)")
    parser.add_argument('--device', type=str, default="cuda:2", help="Which device to run on (cuda:0, cuda:1, cuda:2)")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs for training")
    args = parser.parse_args()
    main(args)       
        