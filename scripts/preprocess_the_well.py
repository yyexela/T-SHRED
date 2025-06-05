# ## Imports

import sys
import torch
import pickle
from tqdm import tqdm
from pathlib import Path

from the_well.data import WellDataset

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root.absolute()))

print("Project root:", project_root)

from src.helpers import *

debug = False
dataset = 'planetswe'
n_steps = {'planetswe': 1008, 'active_matter': 81, 'gray_scott_reaction_diffusion': 1001}
total_tracks = {'planetswe': 10, 'active_matter': 360, 'gray_scott_reaction_diffusion': 1200}
#total_tracks = {'planetswe': 120, 'active_matter': 360, 'gray_scott_reaction_diffusion': 1200}

print(f"Preprocessing {dataset}")

save_dir = project_root / 'datasets' / 'the_well_custom' / dataset
save_dir.mkdir(exist_ok=True, parents=True)

# ## Load Data

# Load data from online, when using in practice we'll have to download the dataset
train_data = WellDataset(
    well_base_path=Path(project_root / 'datasets' / 'the_well'),
    well_dataset_name=dataset,
    well_split_name="train",
    n_steps_input=n_steps[dataset],
    n_steps_output=0,
    use_normalization=False,
)

valid_data = WellDataset(
    well_base_path=Path(project_root / 'datasets' / 'the_well'),
    well_dataset_name=dataset,
    well_split_name="valid",
    n_steps_input=n_steps[dataset],
    n_steps_output=0,
    use_normalization=False,
)

test_data = WellDataset(
    well_base_path=Path(project_root / 'datasets' / 'the_well'),
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

print("full_mat.shape:", full_mat.shape) # (n_steps * n_tracks, im_rows, im_cols, im_dim)

# ## Scale data

dim_scalers = []
full_dim_scaled_l = []

for i in range(im_dim):
    full_dim_scaled, dim_scaler = min_max_scale(full_mat[:,:,:,i])
    dim_scalers.append(dim_scaler)
    full_dim_scaled_l.append(full_dim_scaled)

full_mat_scaled = torch.stack(full_dim_scaled_l, dim=3)

# ## Separate into Training, Validation, and Testing Data

train_num = int(0.8*n_steps[dataset])
test_num = int(0.1*n_steps[dataset])
valid_num = n_steps[dataset] - train_num - test_num

print("train_num:", train_num)
print("valid_num:", valid_num)
print("test_num:", test_num)

# ## Save Results

# Create directories
(save_dir / 'metadata').mkdir(exist_ok=True)
(save_dir / 'full').mkdir(exist_ok=True)
(save_dir / 'pod').mkdir(exist_ok=True)

for track_num in tqdm(range(total_tracks[dataset]), desc="Saving full data"):
    track_start_idx = track_num*n_steps[dataset]
    track_end_idx = (track_num+1)*n_steps[dataset]
    track = full_mat_scaled[track_start_idx:track_end_idx,:] # torch.Size([1008, 393216])

    train = track[0:train_num] # torch.Size([806, 393216])
    valid = track[train_num:train_num+valid_num] # torch.Size([100, 393216])
    test = track[train_num+valid_num:] # torch.Size([102, 393216])

    with open(save_dir / 'full' / f'train_{track_num}.pkl', 'wb') as f:
        pickle.dump(train.numpy(), f)

    with open(save_dir / 'full' / f'valid_{track_num}.pkl', 'wb') as f:
        pickle.dump(valid.numpy(), f)

    with open(save_dir / 'full' / f'test_{track_num}.pkl', 'wb') as f:
        pickle.dump(test.numpy(), f)

# ## Save Metadata

with open(save_dir / 'metadata' / 'scalers.pkl', 'wb') as f:
    pickle.dump(dim_scalers, f)

with open(save_dir / 'metadata' / 'im_dims.pkl', 'wb') as f:
    pickle.dump((im_rows, im_cols, im_dim), f)

