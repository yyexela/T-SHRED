import os
import time
import argparse
from pathlib import Path

top_dir = Path(__file__).parent.parent
slurm_dir = top_dir / 'slurms'
pickle_dir = top_dir / 'pickles'
checkpoint_dir = top_dir / 'checkpoints'

def get_identifier(filename):
    """Extract identifier from filename by removing extension and _test_loss suffix."""
    name = Path(filename).stem  # Remove extension
    return name

def main(args=None):
    """
    Given slurm_dir and pickle_dir, compare all the filenames between the two directories and print the names of the files that are not present in both directories.
    """
    # Get all files from both directories and extract identifiers
    slurm_files = {f.name for f in slurm_dir.glob('*.slurm') if f.is_file()}
    pickle_files = {f.name for f in pickle_dir.glob('*.pkl') if f.is_file()}
    
    # Get sets of identifiers
    slurm_ids = {get_identifier(f) for f in slurm_files}
    pickle_ids = {get_identifier(f) for f in pickle_files}
    
    # Find identifiers that are not present in both directories
    only_in_slurm = slurm_ids - pickle_ids
    only_in_pickle = pickle_ids - slurm_ids
    
    # Print results
    if only_in_slurm:
        print("\nIdentifiers only in slurm directory:")
        for identifier in sorted(only_in_slurm):
            print(f"  {identifier}")
        print()
            
    if only_in_pickle:
        print("\nIdentifiers only in pickle directory:")
        for identifier in sorted(only_in_pickle):
            print(f"  {identifier}")
        print()

    if args.rerun:
        for identifier in only_in_slurm:
            os.system(f"sbatch {slurm_dir / f'{identifier}.slurm'}")

    if args.clean_checkpoint:
        for identifier in pickle_ids:
            try:
                os.remove(checkpoint_dir / f'{identifier}_model_best.pt')
                os.remove(checkpoint_dir / f'{identifier}_model_latest.pt')
            except:
                pass
            
    if not only_in_slurm and not only_in_pickle:
        print("All identifiers are present in both directories.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true', help="Re-run the slurms that are not present in the pickle directory")
    parser.add_argument('--clean_checkpoint', action='store_true', help="Delete checkpoint files that are present in the pickle directory")
    args = parser.parse_args()
    main(args)       
        
