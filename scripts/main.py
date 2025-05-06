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

########
# Main #
########

#validation_errors = models.fit(UTransformer, train_dataset, valid_dataset, batch_size=25, num_epochs=8, lr=0.001, verbose=True, patience=5)
#UTransformer = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)

def main(args=None):
    # Load dataset
    train_dl, valid_dl, test_dl = datasets.load_dataset(args)
    args.n_sensors, args.d_data = (next(iter(train_dl))[0].shape[-2],
                                   next(iter(train_dl))[0].shape[-1])
    args.data_rows, args.data_cols = (next(iter(train_dl))[1].shape[-2],
                                      next(iter(train_dl))[1].shape[-1])
    args.d_model = args.n_sensors * args.d_data
    args.output_size = args.data_rows*args.data_cols*args.d_data

    # Load model
    model = models.MixedModel(args)

    validation_errors = helpers.train_model(model, train_dl, args)
    print(list(validation_errors))

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help="Dataset batch size")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run")
    parser.add_argument('--decoder', type=str, default="mlp", help="Which decoder to use (unet, mlp)")
    parser.add_argument('--decoder_depth', type=int, default=2, help="Number of decoder layers")
    parser.add_argument('--device', type=str, default="cuda:2", help="Which device to run on")
    parser.add_argument('--dim_feedforward', type=int, default=128, help="Size of feed forward layers in transformer encoder")
    parser.add_argument('--dropout', type=float, default=0.1, help="Model droput proportion")
    parser.add_argument('--encoder', type=str, default="sindy_attention_transformer", help="Which encoder to use (transformer, sindy_attention_transformer, sindy_loss_transformer)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--hidden_size', type=int, default=12, help="Hidden size of encoder")
    parser.add_argument('--include_sine', action='store_true', help="Include sine in transformer SINDy library")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument('--n_heads', type=int, default=6, help="Number of transformer heads")
    parser.add_argument('--num_sindy_layers', type=int, default=2, help="Number of SINDY layers")
    parser.add_argument('--poly_order', type=int, default=2, help="Order of polynomial library for SINDy transformer library")
    parser.add_argument('--use_normalization', type=int, default=6, help="Use normalization for datasets")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose messages")
    parser.add_argument('--window_length', type=int, default=10, help="Dataset window length")
    args = parser.parse_args()
    main(args)       
        