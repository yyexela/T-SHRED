import os
import time
from pathlib import Path

top_dir = Path(__file__).parent.parent
slurm_dir = top_dir / 'slurms'

def main(args=None):
    for slurm_file in slurm_dir.glob('*.slurm'):
        print(f"Submitting {slurm_file}")
        os.system(f'sbatch {slurm_file}')
        time.sleep(0.1)
    
if __name__ == '__main__':
    main()       
        
