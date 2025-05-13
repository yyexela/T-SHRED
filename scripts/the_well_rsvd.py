
# ## The Well rSVD Preprocessing
# 
# [link to official documentation](https://polymathic-ai.org/the_well/tutorials/dataset/)  
# [link to original paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4f9a5acd91ac76569f2fe291b1f4772b-Paper-Datasets_and_Benchmarks_Track.pdf)

# ## Imports

import sys
import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from einops import rearrange
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download
from the_well.data import WellDataset
from torch.utils.data import DataLoader

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root.absolute()))

from src.helpers import *

debug = False
n_iters = 5
dataset = 'planetswe'
n_steps = {'planetswe': 1008, 'active_matter': 81, 'gray_scott_reaction_diffusion': 1001}
n_rank = {'active_matter': 50, 'planetswe': 75, 'gray_scott_reaction_diffusion': 75}
total_tracks = {'planetswe': 120, 'active_matter': 360, 'gray_scott_reaction_diffusion': 1200}
#total_tracks = {'planetswe': 24, 'active_matter': 72, 'gray_scott_reaction_diffusion': 240}

save_dir = project_root / 'datasets' / 'the_well_custom' / dataset
save_dir.mkdir(exist_ok=True, parents=True)

# ## Load Data

# Load data from online, when using in practice we'll have to download the dataset
train_data = WellDataset(
    well_base_path=Path('/data') / 'alexey' / 'the_well',
    well_dataset_name=dataset,
    well_split_name="train",
    n_steps_input=n_steps[dataset],
    n_steps_output=0,
    use_normalization=False,
)

valid_data = WellDataset(
    well_base_path=Path('/data') / 'alexey' / 'the_well',
    well_dataset_name=dataset,
    well_split_name="valid",
    n_steps_input=n_steps[dataset],
    n_steps_output=0,
    use_normalization=False,
)

test_data = WellDataset(
    well_base_path=Path('/data') / 'alexey' / 'the_well',
    well_dataset_name=dataset,
    well_split_name="test",
    n_steps_input=n_steps[dataset],
    n_steps_output=0,
    use_normalization=False,
)

im_shape = train_data[0]['input_fields'].shape
im_rows, im_cols, im_dim = im_shape[1], im_shape[2], im_shape[3]
print("im_shape:", im_shape)

# ## Concatenate

full_mat = create_mats_full(train_data, valid_data, test_data, total_tracks[dataset], debug=debug)

print("full_mat.shape:", full_mat.shape)

# ## Save full_mat

# ## Separate into Training, Validation, and Testing Data

train_num = int(0.8*n_steps[dataset])
test_num = int(0.1*n_steps[dataset])
valid_num = n_steps[dataset] - train_num - test_num

print("train_num:", train_num)
print("valid_num:", valid_num)
print("test_num:", test_num)

train_save = []
valid_save = []
test_save = []

for track_num in range(total_tracks[dataset]):
    track = full_mat[track_num*n_steps[dataset]:(track_num+1)*n_steps[dataset],:]

    train = track[0:train_num]
    val = track[train_num:train_num+valid_num]
    test = track[train_num+valid_num:]

    train_save.append(train)
    valid_save.append(val)
    test_save.append(test)

# ## Save Results

# Create directories
(save_dir / 'metadata').mkdir(exist_ok=True)
(save_dir / 'full').mkdir(exist_ok=True)
(save_dir / 'pod').mkdir(exist_ok=True)

# Save full and pod tracks
for i in range(total_tracks[dataset]):
    torch.save(train_save[i], save_dir / 'full' / f'train_{i}.pt')
    torch.save(valid_save[i], save_dir / 'full' / f'valid_{i}.pt')
    torch.save(test_save[i], save_dir / 'full' / f'test_{i}.pt')

del train_save
del valid_save
del test_save

# ## POD

_, _, V_full = generate_SVD(full_mat, n_rank=n_rank[dataset], n_iters=n_iters)

full_pod = create_pod(full_mat, V_full)

del full_mat

print("full_pod.shape:", full_pod.shape)

full_pod_scaled, full_scaler = scale_pod(full_pod)

# ## Separate into Training, Validation, and Testing Data

train_pods_save = []
valid_pods_save = []
test_pods_save = []

for track_num in range(total_tracks[dataset]):
    track_pod_scaled = full_pod_scaled[track_num*n_steps[dataset]:(track_num+1)*n_steps[dataset],:]

    train_pod_scaled = track_pod_scaled[0:train_num]
    valid_pod_scaled = track_pod_scaled[train_num:train_num+valid_num]
    test_pod_scaled = track_pod_scaled[train_num+valid_num:]

    train_pods_save.append(train_pod_scaled)
    valid_pods_save.append(valid_pod_scaled)
    test_pods_save.append(test_pod_scaled)

# ## Save Results

# Create directories
(save_dir / 'metadata').mkdir(exist_ok=True)
(save_dir / 'full').mkdir(exist_ok=True)
(save_dir / 'pod').mkdir(exist_ok=True)

# Save scaler, V_full, and image metadata
torch.save(V_full, save_dir / 'metadata' / 'V.pt')
torch.save(full_scaler, save_dir / 'metadata' / 'scaler.pt')
torch.save((im_rows, im_cols, im_dim), save_dir / 'metadata' / 'im_dims.pt')

# Save full and pod tracks
for i in range(total_tracks[dataset]):
    torch.save(train_pods_save[i], save_dir / 'pod' / f'train_{i}.pt')
    torch.save(valid_pods_save[i], save_dir / 'pod' / f'valid_{i}.pt')
    torch.save(test_pods_save[i], save_dir / 'pod' / f'test_{i}.pt')

del train_pods_save
del valid_pods_save
del test_pods_save

# ## Print Errors
cumulative_errors_numerator = 0.
cumulative_errors_denominator = 0.

for i in range(total_tracks[dataset]):
    train_full = torch.load(save_dir / 'full' / f'train_{i}.pt') 
    valid_full = torch.load(save_dir / 'full' / f'valid_{i}.pt') 
    test_full = torch.load(save_dir / 'full' / f'test_{i}.pt') 

    train_pod = torch.load(save_dir / 'pod' / f'train_{i}.pt') 
    valid_pod = torch.load(save_dir / 'pod' / f'valid_{i}.pt') 
    test_pod = torch.load(save_dir / 'pod' / f'test_{i}.pt') 

    mat_full = torch.cat([train_full, valid_full, test_full], dim=0)
    del train_full
    del valid_full
    del test_full

    mat_pod = torch.cat([train_pod, valid_pod, test_pod], dim=0)
    del train_pod
    del valid_pod
    del test_pod

    mat_pod_hat = inverse_pod(mat_pod, full_scaler, V_full)
    del mat_pod

    cumulative_errors_numerator += (mat_full - mat_pod_hat).pow(2).sum(axis=-1)
    cumulative_errors_denominator += mat_full.pow(2).sum(axis=-1)

    del mat_full
    del mat_pod_hat

cumulative_error = (cumulative_errors_numerator.sqrt() / cumulative_errors_denominator.sqrt()).mean()

print("Total error:", number_to_percentage(cumulative_error))



