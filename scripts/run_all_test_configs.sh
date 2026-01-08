#!/bin/bash
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

