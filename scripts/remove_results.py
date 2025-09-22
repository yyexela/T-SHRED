import yaml
import shutil
from pathlib import Path

top_dir = Path(__file__).parent.parent

results_dir = top_dir / 'results'
config_files = []
for config_file in results_dir.glob('**/*.yaml'):
    if 'optimal_params' in str(config_file):
        config_files.append(config_file)

def extract_seed(config_file):
    filename = config_file.stem
    seed_str = filename.split('_')[-1]
    return int(seed_str)

def extract_identifier(config_file):
    return config_file.stem

def extract_dataset(config_file):
    identifier = extract_identifier(config_file)
    dataset = identifier.split('_')[0]
    return dataset

# Go through config files and removes those which are optimized
delete_count = 0
keep_count = 0
for config_file in config_files:
    # Load yaml file config_file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    identifier = config['model']['identifier']
    dataset = config['model']['dataset']
    if config['model']['forecast_length'] != 1:
        shutil.rmtree(config_file.parent)
        print("Deleting config:", config_file.parent)
        delete_count += 1
    else:
        keep_count += 1

print(f"\nSummary:")
print(f"Total configs: {len(config_files)}")
print(f"Total jobs skipped: {delete_count}")
print(f"Total jobs kept: {keep_count}")

