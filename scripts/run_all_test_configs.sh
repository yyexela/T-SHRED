#!/bin/bash

# This script runs all the test configurations in the `configs/test/` directory.
# Exits as soon as one configuration fails.

set -e

source ~/.virtualenvs/tshred/bin/activate

for config in configs/test/*.yaml; do
    # Skip template.yaml
    if [[ "$(basename "$config")" == "template.yaml" ]]; then
        continue
    fi
    
    echo "=========================================="
    echo "Running config: $config"
    echo "=========================================="
    
    python scripts/main.py --config "$config"
done

echo "=========================================="
echo "All test configs completed successfully!"
echo "=========================================="

