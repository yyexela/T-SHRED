#!/usr/bin/env python3
"""Remove all hp_config*.yaml files from the results/ directory."""

import os
from pathlib import Path


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all hp_config*.yaml files recursively
    pattern = "hp_config*.yaml"
    files_to_remove = list(results_dir.rglob(pattern))

    if not files_to_remove:
        print(f"No files matching '{pattern}' found in {results_dir}")
        return

    print(f"Found {len(files_to_remove)} file(s) to remove:")
    for file_path in files_to_remove:
        print(f"  - {file_path.relative_to(script_dir)}")

    # Remove the files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_count += 1
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

    print(f"\nSuccessfully removed {removed_count} file(s).")


if __name__ == "__main__":
    main()
