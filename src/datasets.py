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

import src.models as models
from src.processdata import load_data
from src.processdata import TimeSeriesDataset

# Directories
top_dir = Path(__file__).parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

#############
# Functions #
#############

def initialize_plasma_data():
    # Load data
    v_total = np.load(plasma_dir / 'v_total.npy') #Data for v location
    s_total = np.load(plasma_dir / 's_total.npy') #Data for s location
    u_total = np.load(plasma_dir / 'u_total.npy') #Data for u location

    plasma_data = sio.loadmat(plasma_dir / 'ne.mat') #Load file from matlab
    utemp = plasma_data['Data'] # Access plasma data using the data key
    X = utemp - np.mean(utemp, axis=0) # Find X by subtracting the data by the mean of their columns (normalization)
    Xnorm =  np.max(np.abs(X)) # Returns the absolute value of each element in X and the maximum value in the data
    X = X/Xnorm # Normalization technique
    #Z score normalization

    # Observe shape of data
    n2 = (X).shape[0]
    m2 = s_total.shape[1]  # svd modes used

    num_sensors = 3 # Determines the number of sensors
    lags = 52 # Time delay of 52

    nx = 257
    ny = 256

    load_X = v_total.T # Transpose v_total 

    sensor_locations_ne = np.random.choice(n2, size=num_sensors, replace=False)
    sensor_locations = [0, 1, 2]

    load_X = np.hstack((X[sensor_locations_ne,:].T,load_X))

    n = (load_X).shape[0]
    m = (load_X).shape[1]

    mask = np.zeros(n2)           
    mask[sensor_locations_ne[0]]=1
    mask[sensor_locations_ne[1]]=1
    mask[sensor_locations_ne[2]]=1

    mask2 = mask.reshape((nx,ny)) 

    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(2, 1, 1)
    plt.imshow(mask2, cmap='gray')

    #plt.savefig('measure.pdf')

    # We now select indices to divide the data into training, validation, and test sets.

    # RECONSTRUCTION MODE
    train_indices = np.random.choice(n - lags, size=500, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    # FORECASTING MODE

    scaler = MinMaxScaler()
    scaler = scaler.fit(load_X[train_indices])
    transformed_X = scaler.transform(load_X)

    ### Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    return train_dataset, valid_dataset, test_dataset
