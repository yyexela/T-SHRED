import os
from pathlib import Path

top_dir = Path(__file__).parent.parent
slurm_dir = top_dir / 'slurms'

for slurm_file in slurm_dir.glob('*.slurm'):
    print(f"Submitting {slurm_file}")
    os.system(f'sbatch {slurm_file}')
