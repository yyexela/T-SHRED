# Imports

import os
import sys
import torch
import einops
import random
import pickle
import numpy as np
import pprint as pp
import torch.nn as nn
from pathlib import Path
from pyfiglet import Figlet
from torch.utils.data import DataLoader
from src.plots import plot_losses, plot_field_comparison

# Add the project root directory to Python path
top_dir = Path('/') / 'home' / 'alexey' / 'Research4' / 'T-SHRED'
sys.path.insert(0, str(top_dir.absolute()))

from src import *

slurm_dir = top_dir / 'slurms'
pickle_dir = top_dir / 'pickles'

# We will iterate through every combination of these
datasets = ["sst", "plasma", "planetswe_pod", "gray_scott_reaction_diffusion_pod"]
encoders = ["lstm", "vanilla_transformer", "sindy_attention_transformer"]
decoders = ["mlp", "cnn"]
lrs = [f"lr{i:0.2e}" for i in [1e-2, 1e-3, 1e-4, 1e-5]]

# These two will be zipped pairwise
encoder_depths = [f"e{i}" for i in[1, 2, 3]]
decoder_depths = [f"d{i}" for i in[1, 2]]


# Get list of results
results = helpers.get_dictionaries_from_pickles(pickle_dir, early_stop=None)

def generate_latex_table(results, dataset_name):
    """
    Generate a LaTeX table from the results dictionary for a specific dataset.
    Shows only the top 10 best performing models.
    
    Args:
        results (list): List of dictionaries containing model results
        dataset_name (str): Name of the dataset to filter results for
        
    Returns:
        str: LaTeX table as a string
    """
    # Filter results for the specified dataset
    dataset_results = [r for r in results if r['hyperparameters']['dataset'] == dataset_name]
    
    # Sort results by test loss and take top 10
    sorted_results = sorted(dataset_results, key=lambda x: x['test_loss'])[:10]
    
    # Start building the LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "Encoder & Decoder & Enc. Layers & Learning Rate & Test Loss \\\\\n"
    latex += "\\hline\n"
    
    for result in sorted_results:
        hp = result['hyperparameters']
        latex += f"{hp['encoder']} & {hp['decoder']} & {hp['encoder_depth']} & {hp['lr']} & {result['test_loss']:.4e} \\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += f"\\caption{{Top 10 best performing models for {dataset_name} dataset}}\n"
    latex += f"\\label{{tab:model_results_{dataset_name}}}\n"
    latex += "\\end{table}"
    
    return latex

# Example usage for each dataset
for dataset in ["sst", "plasma", "planetswe_full"]:
    print(f"\nResults for {dataset} dataset:")
    latex_table = generate_latex_table(results, dataset)
    print(latex_table)
    print("\n" + "="*80 + "\n")

pass

