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
from torch.utils.data import Dataset

from the_well.data import WellDataset

# Local files
pkg_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(pkg_path))

from src.helpers import min_max_scale, get_dataset_dims

# Directories
top_dir = Path(__file__).parent.parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

#############
# Functions #
#############

class TimeSeriesDataset(Dataset):
    def __init__(self, input_tensors, window_length, device='cpu', output_tensors=None):
        """
        Args:
            input_tensors (list of torch.Tensor): List of input tensors where each tensor is
                a time series of shape (time_steps, features)
            output_tensors (list of torch.Tensor): List of output tensors where each tensor is
                a time series of shape (time_steps, features). When None, use input_tensors as output_tensors.
            window_length (int): Length of the sliding window
            device (str): Device to move the tensors to
        """
        super().__init__()
        self.window_length = window_length

        self.input_tensors = input_tensors
        if output_tensors is not None:
            self.output_tensors = output_tensors
        else:
            self.output_tensors = None

        # Convert tensors to torch
        if isinstance(self.input_tensors[0], np.ndarray):
            self.input_tensors = [torch.from_numpy(tensor) for tensor in self.input_tensors]
        if self.output_tensors is not None:
            if isinstance(self.output_tensors[0], np.ndarray):
                self.output_tensors = [torch.from_numpy(tensor) for tensor in self.output_tensors]

        # Float32 for both sensors and tensors
        self.input_tensors = [tensor.float() for tensor in self.input_tensors]
        if self.output_tensors is not None:
            self.output_tensors = [tensor.float() for tensor in self.output_tensors]

        # Move to GPU
        self.input_tensors = [tensor.to(device) for tensor in self.input_tensors]
        if self.output_tensors is not None:
            self.output_tensors = [tensor.to(device) for tensor in self.output_tensors]

        # Calculate cumulative window counts for index mapping
        self.cumulative_offsets = [0]
        current = 0
        for tensor in self.input_tensors:
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

        input_tensor = self.input_tensors[tensor_idx]

        if self.output_tensors is not None:
            output_tensor = self.output_tensors[tensor_idx]
        else:
            output_tensor = input_tensor

        window = input_tensor[start:end]
        target = output_tensor[target].unsqueeze(0)

        return {"input_fields": window, "output_fields": target}

def load_dataset(args):
    if args.dataset == 'sst':
        return load_sst_data(args)
    elif args.dataset == 'sst_demo':
        return load_sst_demo_data(args)
    elif args.dataset == 'plasma':
        return load_plasma_data(args)
    elif args.dataset in ["planetswe"]:
        return load_well_data(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

def load_the_well_pts(load_path, split_name, track_id=None, n_tracks=None):
    tensors = []
    if track_id is not None:
        iter_l = [Path(load_path) / f"{split_name}_{track_id}.pkl"]
    else:
        iter_l = sorted(load_path.iterdir())
    for pt_file in iter_l:
        if split_name in pt_file.name:
            # Convert data to pytorch (treat it like a (1 x dim x 1) image)
            with open(pt_file, 'rb') as f:
                tensor = pickle.load(f)
                tensor = torch.from_numpy(tensor)
            tensor = tensor.float()
            tensors.append(tensor)
            if len(tensors) >= n_tracks:
                break
    return tensors

def load_well_data(args):
    # Data path
    data_path = data_dir / 'the_well_custom' / args.dataset

    # Load training, validation, and testing data
    train_fulls = load_the_well_pts(data_path / 'full', 'train', n_tracks=args.n_well_tracks)
    val_fulls = load_the_well_pts(data_path / 'full', 'valid', n_tracks=args.n_well_tracks)
    test_fulls = load_the_well_pts(data_path / 'full', 'test', n_tracks=args.n_well_tracks)

    # Load Scalers
    with open(data_path / 'metadata' / 'scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)

    # Create torch datasets
    train_full_ds = TimeSeriesDataset(input_tensors=train_fulls, window_length=args.window_length, device=args.device)
    valid_full_ds = TimeSeriesDataset(input_tensors=val_fulls, window_length=args.window_length, device=args.device)
    test_full_ds = TimeSeriesDataset(input_tensors=test_fulls, window_length=args.window_length, device=args.device)

    return train_full_ds, valid_full_ds, test_full_ds, {'scalers': scalers}

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
    _, scaler = min_max_scale(sst_data)
    train, _ = min_max_scale(train, scaler=scaler)
    val, _ = min_max_scale(val, scaler=scaler)
    test, _ = min_max_scale(test, scaler=scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(input_tensors=[split], window_length=args.window_length, device=args.device)
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, {'scalers': [scaler]}

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
    _, scaler = min_max_scale(sst_data)
    train, _ = min_max_scale(train, scaler=scaler)
    val, _ = min_max_scale(val, scaler=scaler)
    test, _ = min_max_scale(test, scaler=scaler)

    # Create torch datasets
    datasets = []
    for i, split in enumerate([train, val, test]):
        sst_ds = TimeSeriesDataset(input_tensors=[split], window_length=args.window_length, device=args.device)
        datasets.append(sst_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, {'scalers': [scaler]}

def load_plasma_data(args):
    # Load data
    ne_data = sio.loadmat(plasma_dir / 'ne.mat') # (65792, 2000) = (257 * 256, 2000)
    ne_data = ne_data['Data']

    u_total = np.load(plasma_dir / 'u_total.npy') # (65792, 280)
    s_total = np.load(plasma_dir / 's_total.npy') # (14, 20)
    v_total = np.load(plasma_dir / 'v_total.npy') # (280, 2000)

    # Switch from ne = U S V to ne * = V* S* U*
    ne_data = ne_data.T # (2000, 257 * 256) = (2000, 65792)
    u_total = u_total.T # (280, 2000)
    s_total = s_total.T # (20, 14)
    v_total = v_total.T # (2000, 280)

    # Convert ne_data to image (2000, 257, 256, 1)
    ne_data = einops.rearrange(ne_data, "t (r w d) -> t r w d", t=ne_data.shape[0], r=257, w=256, d=1)

    # Convert v_total output to image (2000, 1, 280, 1)
    v_total_output = einops.rearrange(v_total, "t (r w d) -> t r w d", t=v_total.shape[0], r=1, w=280, d=1)

    # Create training, testing, and validation split
    train_size = int(ne_data.shape[0] * 0.8)
    val_size = int(ne_data.shape[0] * 0.1)
    input_train, input_val, input_test = np.split(ne_data, [train_size, train_size + val_size])
    output_train, output_val, output_test = np.split(v_total_output, [train_size, train_size + val_size])

    # Convert data to pytorch
    input_train = torch.from_numpy(input_train).float()
    input_val = torch.from_numpy(input_val).float()
    input_test = torch.from_numpy(input_test).float()

    output_train = torch.from_numpy(output_train).float()
    output_val = torch.from_numpy(output_val).float()
    output_test = torch.from_numpy(output_test).float()

    # Min Max Scale input data
    # Note: u_total and v_total are all in [-1,1] already, s_total is eigenvalues and has a larger range but we can't do anything about that
    _, scaler = min_max_scale(ne_data)
    input_train, _ = min_max_scale(input_train, scaler=scaler)
    input_val, _ = min_max_scale(input_val, scaler=scaler)
    input_test, _ = min_max_scale(input_test, scaler=scaler)

    # Create torch datasets
    datasets = []
    for i, (input_split, output_split) in enumerate([(input_train, output_train),
                                                     (input_val, output_val),
                                                     (input_test, output_test)]):
        plasma_ds = TimeSeriesDataset(input_tensors=[input_split], output_tensors=[output_split], window_length=args.window_length, device=args.device)
        datasets.append(plasma_ds)

    train_ds = datasets[0]
    valid_ds = datasets[1]
    test_ds = datasets[2]

    return train_ds, valid_ds, test_ds, {'scalers': [scaler], 'u_total': u_total, 's_total': s_total, 'v_total': v_total}

