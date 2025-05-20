import os
import time
import argparse
from pathlib import Path

top_dir = Path(__file__).parent.parent
slurm_dir = top_dir / 'slurms'
pickle_dir = top_dir / 'pickles'
checkpoint_dir = top_dir / 'checkpoints'
log_dir = top_dir / 'logs'

def main(args=None):
    """
    Rename files in pickles and checkpoints directories by appending 'p2' to filenames.
    For checkpoints, append 'p2' before '_model_best' and '_model_latest'.
    """
    # Rename pickle files
    for file in pickle_dir.glob('*.pkl'):
        new_name = file.with_name(f"{file.stem}_p2{file.suffix}")
        #print(f"Renamed pickle: {file.name} -> {new_name.name}")
        file.rename(new_name)

    # Rename checkpoint files
    for file in checkpoint_dir.glob('*'):
        if file.is_file():
            name = file.name
            if '_model_best' in name:
                new_name = name.replace('_model_best', '_p2_model_best')
            elif '_model_latest' in name:
                new_name = name.replace('_model_latest', '_p2_model_latest')
            else:
                continue
            
            new_path = file.parent / new_name
            #print(f"Renamed checkpoint: {file.name} -> {new_name}")
            file.rename(new_path)

    # Rename log files
    for file in log_dir.glob('*'):
        if file.is_file():
            name = file.stem
            #print(name)

            identifier = name[-8:]
            name = name[:-8]

            new_name = f'{name}p2_{identifier}{file.suffix}'

            new_path = file.parent / new_name
            #print(f"Renamed checkpoint: {file.name} -> {new_name}")
            file.rename(new_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    main(args)       
        
