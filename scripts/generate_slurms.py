from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={slurm_time}
#SBATCH --nice=0

#SBATCH --job-name="{model_name}_{identifier}"
#SBATCH --output=/mmfs1/home/alexeyy/storage/T-SHRED/logs/{model_name}/{identifier}_%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=alexeyy@uw.edu

repo="/mmfs1/home/alexeyy/storage/T-SHRED"

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /mmfs1/home/alexeyy/.virtualenvs/tshred/bin/activate
mkdir -p $repo/logs/{model_name}
python -u $repo/scripts/hyper_opt.py --model-name {model_name} --config $repo/configs/{model_name}/tuning_config/{identifier}.yaml --time-budget-hours {hyper_opt_time} --use-asha --gpus-per-trial 1 --log-to-file --log-dir $repo/logs/{model_name}

echo "Finished running Python"\
"""

# Slurm parameters
account='amath'
partition='gpu-rtx6k'
memory=32
cpus_per_task=12
slurm_time='6:00:00'
hyper_opt_time=5.9

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

# Recursively find all tuning configs in configs directory only if 'tuning_config' is a part of the path
configs_dir = top_dir / 'configs'
config_files = []
for config_file in configs_dir.glob('**/*.yaml'):
    if 'tuning_config' in str(config_file):
        config_files.append(config_file)

for config_file in config_files:
    config_file_name = config_file.name
    model_name = config_file.parent.parent.name
    identifier = config_file_name.split('.')[0]

    cmd = cmd_template.format(
        account=account,
        partition=partition,
        memory=memory,
        cpus_per_task=cpus_per_task,
        slurm_time=slurm_time,
        hyper_opt_time=hyper_opt_time,
        model_name=model_name,
        identifier=identifier,
    )

    with open(top_dir / 'slurms' / f'{model_name}_{identifier}.slurm', "w") as f:
        f.write(cmd)

print(f"Total jobs: {len(config_files)}")
