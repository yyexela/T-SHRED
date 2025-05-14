###########
# Imports #
###########

import sys
import gzip
import torch
import bisect
import pickle
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
from src.helpers import min_max_scale

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
    elif args.dataset == 'sst_demo':
        return load_sst_demo_data(args)
    elif args.dataset == 'plasma':
        return load_plasma_data(args)
    elif args.dataset in ["active_matter", "planetswe"]:
        return load_well_data(args)
    elif args.dataset in ["active_matter_pod", "planetswe_pod"]:
        return load_well_data_pod(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

def load_well_data(args):
    train_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="train",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=True,
    )

    valid_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="valid",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=True,
    )

    test_dl = WellDataset(
        well_base_path=data_dir / 'the_well',
        well_dataset_name=args.dataset,
        well_split_name="test",
        n_steps_input=args.window_length,
        n_steps_output=1,
        use_normalization=True,
    )

    return train_dl, valid_dl, test_dl, (None, None)

def load_the_well_pts(load_path, split_name, reshape_to_image=False):
    tensors = []
    for pt_file in sorted(load_path.iterdir()):
        if split_name in pt_file.name:
            # Convert data to pytorch (treat it like a (1 x dim x 1) image)
            with open(pt_file, 'rb') as f:
                tensor = pickle.load(f)
                tensor = torch.from_numpy(tensor)
            tensor = tensor.float()
            if reshape_to_image:
                tensor = tensor.unsqueeze(1).unsqueeze(-1)
            tensors.append(tensor)
    return tensors

def load_well_data_pod(args):
    # Data path
    data_path = data_dir / 'the_well_custom' / args.dataset[:-4]

    # Load training, validation, and testing data
    train_pods = load_the_well_pts(data_path / 'pod', 'train', reshape_to_image=True)
    val_pods = load_the_well_pts(data_path / 'pod', 'valid', reshape_to_image=True)
    test_pods = load_the_well_pts(data_path / 'pod', 'test', reshape_to_image=True)

    if args.eval_full:
        train_fulls = load_the_well_pts(data_path / 'full', 'train')
        val_fulls = load_the_well_pts(data_path / 'full', 'valid')
        test_fulls = load_the_well_pts(data_path / 'full', 'test')

    # Load V and scalers
    with open(data_path / 'metadata' / 'V.pkl', 'rb') as f:
        V = pickle.load(f)
        V = torch.from_numpy(V)
    with open(data_path / 'metadata' / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(data_path / 'metadata' / 'im_dims.pkl', 'rb') as f:
        im_dims = pickle.load(f)

    # Create torch datasets
    train_pod_ds = TimeSeriesDataset(tensors=train_pods, window_length=args.window_length)
    valid_pod_ds = TimeSeriesDataset(tensors=val_pods, window_length=args.window_length)
    test_pod_ds = TimeSeriesDataset(tensors=test_pods, window_length=args.window_length)

    if args.eval_full:
        train_full_ds = TimeSeriesDataset(tensors=train_fulls, window_length=args.window_length)
        valid_full_ds = TimeSeriesDataset(tensors=val_fulls, window_length=args.window_length)
        test_full_ds = TimeSeriesDataset(tensors=test_fulls, window_length=args.window_length)

    if args.eval_full:
        return train_pod_ds, valid_pod_ds, test_pod_ds, train_full_ds, valid_full_ds, test_full_ds, (V, scaler, im_dims)
    else:
        return train_pod_ds, valid_pod_ds, test_pod_ds, train_full_ds, valid_full_ds, test_full_ds, (V, scaler, im_dims)

def load_sst_data(args):
    # Load raw file
    sst_data_path = data_dir / 'sst' / "SST_data.mat"
    sst_data = sio.loadmat(sst_data_path)['Z'] # (64800, 1400)
    sst_data = einops.rearrange(sst_data, '(r c) t -> t r c', r=180, c=360, t=1400)

    # Create training, testing, and validation split
    train_size = int(sst_data.shape[0] * 0.8)
    val_size = int(sst_data.shape[0] * 0.1)
    train, val, test = np.split(sst_data, [train_size, train_size + val_size])

    # Convert data to pytorch (treat it like a row x col x 1 image)
    train = torch.from_numpy(train).float().unsqueeze(-1)
    val = torch.from_numpy(val).float().unsqueeze(-1)
    test = torch.from_numpy(test).float().unsqueeze(-1)

    # Min Max Scale data
    train, scaler = min_max_scale(train)
    val, _ = min_max_scale(val, scaler)
    test, _ = min_max_scale(test, scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(tensors=[split], window_length=args.window_length)
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, scaler

def load_sst_demo_data(args):
    # Load raw file
    sst_data_path = data_dir / 'sst' / "demo_sst.npy.gz"
    with gzip.open(sst_data_path, 'rb') as f:
        sst_data = np.load(f) # (1000, 180, 360)

    # Create training, testing, and validation split
    train_size = int(sst_data.shape[0] * 0.8)
    val_size = int(sst_data.shape[0] * 0.1)
    train, val, test = np.split(sst_data, [train_size, train_size + val_size])

    # Convert data to pytorch (treat it like a row x col x 1 image)
    train = torch.from_numpy(train).float().unsqueeze(-1)
    val = torch.from_numpy(val).float().unsqueeze(-1)
    test = torch.from_numpy(test).float().unsqueeze(-1)

    # Min Max Scale data
    train, scaler = min_max_scale(train)
    val, _ = min_max_scale(val, scaler)
    test, _ = min_max_scale(test, scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(tensors=[split], window_length=args.window_length)
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, scaler

def load_plasma_data(args):
    # Load data
    plasma_data = sio.loadmat(plasma_dir / 'ne.mat') #Load file from matlab
    plasma_data = plasma_data['Data'] # Access plasma data using the data key

    # Create training, testing, and validation split
    train_size = int(plasma_data.shape[0] * 0.8)
    val_size = int(plasma_data.shape[0] * 0.1)
    train, val, test = np.split(plasma_data, [train_size, train_size + val_size])

    # Convert data to pytorch (treat it like a 1 x 2000 x 1 image)
    train = torch.from_numpy(train).float().unsqueeze(-1).unsqueeze(1)
    val = torch.from_numpy(val).float().unsqueeze(-1).unsqueeze(1)
    test = torch.from_numpy(test).float().unsqueeze(-1).unsqueeze(1)

    # Min Max Scale data
    train, scaler = min_max_scale(train)
    val, _ = min_max_scale(val, scaler)
    test, _ = min_max_scale(test, scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        plasma_ds = TimeSeriesDataset(tensors=[split], window_length=args.window_length)
        datasets.append(plasma_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, scaler
