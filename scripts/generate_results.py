###########
# Imports #
###########

import sys
import torch
import argparse
import pprint as pp
from pathlib import Path

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
    results = helpers.get_top_N_models_by_loss(args.dataset, pickle_dir)
    pp.pprint(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to run (gray_scott_reaction_diffusion_pod, planetswe_pod, sst, sst_demo, plasma)")
    args = parser.parse_args()
    main(args)       
        
