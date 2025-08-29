from pathlib import Path

top_dir = Path(__file__).parent.parent

bash_template_0 = \
"""\
repo="/home/alexey/Git/T-SHRED"

# Create logs directory and set up logging
mkdir -p $repo/logs
exec > >(tee $repo/logs/{log_filename}) 2>&1

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/alexey/.virtualenvs/tshred/bin/activate
"""

bash_template_1 = \
"""\
mkdir -p $repo/logs/{model_name}

python -u $repo/scripts/hyper_opt.py --config-path $repo/configs/{model_name}/tuning_config/{identifier}.yaml --time-budget-hours {hyper_opt_time} --device {device} --use-asha --gpus-per-trial 0 --log-to-file --log-dir $repo/logs/{model_name}
"""

bash_template_2 = \
"""\
echo "Finished running Python"\
"""

# Parameters
hyper_opt_time = 9999999.0
n_parallel = 2

# Create and clean up bash repo
bash_dir = top_dir / 'bash'
bash_dir.mkdir(exist_ok=True)
for file in bash_dir.glob('*.sh'):
    file.unlink()

# Recursively find all tuning configs in configs directory only if 'tuning_config' is a part of the path
configs_dir = top_dir / 'configs'
config_files = []
for config_file in configs_dir.glob('**/*.yaml'):
    if 'tuning_config' in str(config_file):
        config_files.append(config_file)

device_counter = 0
devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
total_scripts = len(devices) * n_parallel

# Initialize bash scripts for each device and parallel index
bash_scripts = {}
for device in devices:
    for parallel_idx in range(n_parallel):
        script_key = f"{device}_{parallel_idx}"
        device_num = device.split(':')[1]
        log_filename = f"run_cuda_{device_num}_{parallel_idx}.log"
        bash_scripts[script_key] = bash_template_0.format(log_filename=log_filename)

for config_file in config_files:
    config_file_name = config_file.name
    model_name = config_file.parent.parent.name
    identifier = config_file_name.split('.')[0]

    # Determine which device and parallel script to use based on counter
    device_idx = device_counter % len(devices)
    parallel_idx = (device_counter // len(devices)) % n_parallel
    current_device = devices[device_idx]
    script_key = f"{current_device}_{parallel_idx}"
    
    cmd = bash_template_1.format(
        hyper_opt_time=hyper_opt_time,
        model_name=model_name,
        identifier=identifier,
        device=current_device,
    )

    # Add the command to the appropriate bash script
    bash_scripts[script_key] += cmd

    device_counter += 1

# Add the closing template to each script and write to files
for script_key, script_content in bash_scripts.items():
    script_content += bash_template_2
    
    # Parse device and parallel index from script_key
    device, parallel_idx = script_key.rsplit('_', 1)
    device_num = device.split(':')[1]  # Extract number from "cuda:X"
    filename = f"run_cuda_{device_num}_{parallel_idx}.sh"
    filepath = bash_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    filepath.chmod(0o755)
    
    print(f"Generated bash script: {filepath}")

print(f"Total jobs: {len(config_files)}")
print(f"Total scripts generated: {total_scripts}")
print(f"Jobs per script: ~{len(config_files) // total_scripts} (with remainder distributed)")
