import sys
import yaml
import shutil
from pathlib import Path

top_dir = Path(__file__).parent.parent

sys.path.insert(0, str(top_dir))

from src.helpers import extract_config_value, sort_bash_config_key

results_dir = top_dir / 'results'
config_files = []
for config_file in results_dir.glob('**/*.yaml'):
    if 'optimal_params' in str(config_file):
        config_files.append(config_file)

# Go through config files and removes those which are optimized
delete_count = 0
keep_count = 0
for config_file in config_files:
    # Load yaml file config_file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    identifier = extract_config_value(config_file, 'identifier')
    dataset = extract_config_value(config_file, 'dataset')
    if "transformer" in config['model']['encoder']:
        print("Deleting config:", config_file.parent)
        shutil.rmtree(config_file.parent)
        delete_count += 1
    else:
        keep_count += 1

print(f"\nSummary:")
print(f"Total configs: {len(config_files)}")
print(f"Total jobs skipped: {delete_count}")
print(f"Total jobs kept: {keep_count}")

