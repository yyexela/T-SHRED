import sys
from pathlib import Path

top_dir = Path(__file__).parent.parent
sys.path.insert(0, str(top_dir))

from src.helpers import extract_config_value, sort_bash_config_key

bash_template_0 = """\
repo="{repo_path}"

# Create logs directory and set up logging
mkdir -p $repo/logs
exec > >(tee -a $repo/logs/{log_filename}) 2>&1

echo "Running Python on {computer_name}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source {venv_path}
"""

bash_template_1 = """\
CUDA_VISIBLE_DEVICES={device} python -u $repo/scripts/main.py --config $repo/results/{identifier}/optimal_params_{dataset}.yaml
"""

bash_template_2 = """\
echo "Finished running Python"\
"""

# Parameters
n_parallel = 1

# Computer configuration - each computer has its own repo path, venv path, and available GPUs
computers = {
    "computer0": {
        "repo_path": "/home/alexey/Git/T-SHRED",
        "venv_path": "/home/alexey/.virtualenvs/tshred/bin/activate",
        "gpus": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    },
    "computer1": {
        "repo_path": "/home/alexey/Git/T-SHRED",
        "venv_path": "/home/alexey/.virtualenvs/tshred/bin/activate",
        "gpus": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    },
}

# Create and clean up bash repo
bash_dir = top_dir / "bash"
bash_dir.mkdir(exist_ok=True)
for file in bash_dir.glob("*.sh"):
    file.unlink()

# Recursively find all tuning configs in results directory only if 'optimal_params' is a part of the path
configs_dir = top_dir / "results"
config_files = []
for config_file in configs_dir.glob("**/*.yaml"):
    if "optimal_params" in str(config_file):
        config_files.append(config_file)

config_files.sort(key=sort_bash_config_key)

# Build flat list of all available devices across all computers
all_devices = []
for computer_name, computer_config in computers.items():
    for gpu in computer_config["gpus"]:
        all_devices.append((computer_name, gpu))

total_scripts = len(all_devices) * n_parallel

# Initialize bash scripts for each computer-device-parallel combination
bash_scripts = {}
for computer_name, device in all_devices:
    computer_config = computers[computer_name]
    for parallel_idx in range(n_parallel):
        script_key = f"{computer_name}_{device}_{parallel_idx}"
        device_num = device.split(":")[1]
        log_filename = f"run_{computer_name}_cuda_{device_num}_{parallel_idx}.log"
        bash_scripts[script_key] = bash_template_0.format(
            repo_path=computer_config["repo_path"],
            venv_path=computer_config["venv_path"],
            computer_name=computer_name,
            log_filename=log_filename,
        )

# Go through config files and filter out those which are evaluated
config_files_to_process = []

skipped_count = 0
for config_file in config_files:
    identifier = extract_config_value(config_file, "identifier")
    dataset = extract_config_value(config_file, "dataset")
    results_path = Path("/") / "home" / "alexey" / "Git" / "T-SHRED" / "pickles"
    results_path = results_path / f"{identifier}.pkl"

    if not results_path.exists():
        config_files_to_process.append(config_file)
    else:
        skipped_count += 1

# Update config_files to only include files that need processing
config_files = config_files_to_process

if len(config_files) == 0:
    print("No configs to process")
    exit(0)

# Write bash scripts for non-optimized configs
script_written_count = {script_key: 0 for script_key in bash_scripts.keys()}
written_count = 0
device_counter = 0
for config_file in config_files:
    identifier = extract_config_value(config_file, "identifier")
    dataset = extract_config_value(config_file, "dataset")

    # Determine which computer-device-parallel combination to use
    device_idx = device_counter % len(all_devices)
    parallel_idx = (device_counter // len(all_devices)) % n_parallel
    computer_name, current_device = all_devices[device_idx]
    script_key = f"{computer_name}_{current_device}_{parallel_idx}"

    cmd = bash_template_1.format(
        identifier=identifier, dataset=dataset, device=current_device.split(":")[1]
    )

    # Add the command to the appropriate bash script
    bash_scripts[script_key] += cmd

    script_written_count[script_key] += 1

    device_counter += 1
    written_count += 1

# Add the closing template to each script and write to files
for script_key, script_content in bash_scripts.items():
    script_content += bash_template_2

    # Parse computer name, device, and parallel index from script_key
    parts = script_key.split("_")
    computer_name = parts[0]
    device = "_".join(parts[1:-1])  # Handle cuda:X format
    parallel_idx = parts[-1]
    device_num = device.split(":")[1]  # Extract number from "cuda:X"

    filename = f"run_{computer_name}_cuda_{device_num}_{parallel_idx}.sh"
    filepath = bash_dir / filename

    if script_written_count[script_key] > 0:
        with open(filepath, "w") as f:
            f.write(script_content)

        # Make the script executable
        filepath.chmod(0o755)

    print(f"Generated bash script: {filepath}")

# Modify total_scripts to only include scripts with commands
total_scripts = sum(
    1 for script_key in bash_scripts.keys() if script_written_count[script_key] > 0
)

print(f"\nSummary:")
print(f"Total computers: {len(computers)}")
print(f"Total GPUs across all computers: {len(all_devices)}")
print(f"Total jobs skipped: {skipped_count}")
print(f"Total jobs written: {written_count}")
print(f"Total scripts generated: {total_scripts}")
print(
    f"Jobs per script: ~{len(config_files) // total_scripts if total_scripts > 0 else 0} (with remainder distributed)"
)

# Print computer breakdown
for computer_name, computer_config in computers.items():
    computer_gpus = len(computer_config["gpus"])
    computer_scripts = computer_gpus * n_parallel
    print(f"  {computer_name}: {computer_gpus} GPUs, {computer_scripts} scripts")
