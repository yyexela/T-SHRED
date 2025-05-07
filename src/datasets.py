###########
# Imports #
###########

import sys
import gzip
import torch
import bisect
import einops
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from the_well.data import WellDataset

# Local files
pkg_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(pkg_path))

import src.models as models
from src.processdata import load_data
from src.processdata import TimeSeriesDataset
from src.helpers import normalize_pytorch

# Directories
top_dir = Path(__file__).parent.parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

#############
# Functions #
#############

class TimeSeriesDataset(Dataset):
    def __init__(self, tensors, window_length):
        """
        Args:
            tensors (list of torch.Tensor): List of tensors where each tensor is
                a time series of shape (time_steps, features)
            target (str): Column to target as the output
            window_length (int): Length of the sliding window
        """
        super().__init__()
        self.window_length = window_length
        self.tensors = tensors

        # Convert tensors to torch
        if isinstance(self.tensors[0], np.ndarray):
            self.tensors = [torch.from_numpy(tensor) for tensor in self.tensors]

        # Float32 for both sensors and tensors
        self.tensors = [tensor.float() for tensor in self.tensors]

        # Calculate cumulative window counts for index mapping
        self.cumulative_offsets = [0]
        current = 0
        for tensor in self.tensors:
            L = tensor.size(0)
            n_windows = (L - window_length - 1) + 1
            n_windows = max(n_windows, 0)  # Ensure non-negative
            current += n_windows
            self.cumulative_offsets.append(current)
        
        if self.cumulative_offsets[-1] == 0:
            raise ValueError("No valid windows created. Check window_length and tensor lengths.")

    def __len__(self):
        return self.cumulative_offsets[-1]

    def __getitem__(self, idx):
        # Find which tensor contains this index
        tensor_idx = bisect.bisect_right(self.cumulative_offsets, idx) - 1
        
        # Calculate local index within the tensor
        start_idx = self.cumulative_offsets[tensor_idx]
        local_idx = idx - start_idx
        
        # Get corresponding tensor and calculate window
        start = local_idx
        end = start + self.window_length
        target = end # because end is exclusive

        tensor = self.tensors[tensor_idx]

        window = tensor[start:end]
        target = tensor[target].unsqueeze(0)

        return {"input_fields": window, "output_fields": target}

def load_dataset(args):
    if args.dataset == 'sst':
        return load_sst_data(args)
    elif args.dataset == 'plasma':
        return load_plasma_data(args)
    elif args.dataset in ["active_matter", "planetswe"]:
        return load_well_data(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

def load_well_data(args):
    train_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="train",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=args.use_normalization,
    )

    valid_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="valid",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=args.use_normalization,
    )

    test_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="test",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=args.use_normalization,
    )

    return train_dl, valid_dl, test_dl, None

def load_sst_data(args):
    # Load raw file
    sst_data_path = data_dir / 'sst' / "demo_sst.npy.gz"
    with gzip.open(sst_data_path, 'rb') as f:
        sst_data = np.load(f) # (1000, 180, 360)
    sst_data = sst_data # 1 channel

    # Create training, testing, and validation split
    train_size = int(sst_data.shape[0] * 0.8)
    val_size = int(sst_data.shape[0] * 0.1)
    train, val, test = np.split(sst_data, [train_size, train_size + val_size])

    # Convert data to pytorch
    train = torch.from_numpy(train).float().unsqueeze(-1)
    val = torch.from_numpy(val).float().unsqueeze(-1)
    test = torch.from_numpy(test).float().unsqueeze(-1)

    # Normalize data
    train, mean, std = normalize_pytorch(train, (0, 1, 2))
    val, _, _ = normalize_pytorch(val, (0, 1, 2), mean, std)
    test, _, _ = normalize_pytorch(test, (0, 1, 2), mean, std)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(tensors=[split], window_length=args.window_length)
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, (mean, std)

def load_plasma_data(args):
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

    scaler = StandardScaler()
    transformed_X = scaler.fit_transform(load_X[train_indices])

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
