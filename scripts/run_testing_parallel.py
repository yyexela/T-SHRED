import sys
from pathlib import Path

top_dir = Path(__file__).parent.parent
sys.path.insert(0, str(top_dir))

from src.helpers import get_testing_configs, create_semaphores, run_in_parallel

# Parameters
n_parallel = 2

# Computer configuration - each computer has its own repo path, venv path, and available GPUs
computers = {
    "vector": {
        "repo_path": "/home/alexey/Git/T-SHRED",
        "venv_path": "/home/alexey/.virtualenvs/tshred/bin/activate",
        "log_path": "/home/alexey/Git/T-SHRED/logs",
        "gpus": ["cuda:1"]
    },
    #"matrix": {
        #"repo_path": "/home/alexey/Git/T-SHRED",
        #"venv_path": "/home/alexey/.virtualenvs/tshred/bin/activate",
        #"log_path": "/home/alexey/Git/T-SHRED/logs",
        #"gpus": ["cuda:1", "cuda:2", "cuda:3"]
    #},
}

remote_cmd_template = (
    "cd {repo_path} && "
    "source {venv_path} && "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "CUDA_VISIBLE_DEVICES={device_num} "
    "python -u {repo_path}/scripts/main.py --config {config_file}"
)

config_files = get_testing_configs(top_dir)

if len(config_files) == 0:
    print("No configurations to process")
    exit(0)

semaphores = create_semaphores(computers, n_parallel, remote_cmd_template, "testing")
num_devices = sum(len(computer_config["gpus"]) for computer_config in computers.values())

run_in_parallel(config_files, semaphores, n_parallel)

print(f"\nExecution Summary:")
print(f"Total computers: {len(computers)}")
print(f"Total GPUs across all computers: {num_devices}")
print(f"Parallel jobs per GPU: {n_parallel}")
print(f"Total jobs executed: {len(config_files)}")
