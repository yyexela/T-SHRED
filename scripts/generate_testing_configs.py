"""
This script generates configuration files for testing the models.
Writes configuration files into the `configs/test/` directory.
These configurations are used by `run_all_test_configs.sh` to run all the test configurations.

Effectively just a functional verification test to ensure code isn't breaking.
"""

import copy
import yaml
from pathlib import Path

n_seeds = 1
encoders = [
    "gru",
    "lstm",
    "moe_gru",
    "moe_lstm",
    "sindy_loss_gru",
    "sindy_loss_lstm",
    "vanilla_transformer",
    "sindy_loss_transformer",
    "sindy_attention_transformer",
    "sindy_attention_sindy_loss_transformer",
]  # , "sindy_attention_transformer_5", "sindy_attention_sindy_loss_transformer_5"]
decoders = ["mlp", "cnn"]
datasets = ["sst"]

top_dir = Path(__file__).parent.parent


def main():
    # Create output directory and save config, clean up old configs
    output_dir = top_dir / "configs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template config
    template_path = top_dir / "configs/template/test_template.yaml"
    with open(template_path, "r") as f:
        template_config = yaml.safe_load(f)

    seeds = range(n_seeds)

    # Iterate over seeds
    for seed in seeds:
        # Iterate over encoder types
        for encoder in encoders:
            # Iterate over decoder types
            for decoder in decoders:
                # Iterate over datasets
                for dataset in datasets:
                    # Create a deep copy of the template config
                    config = copy.deepcopy(template_config)

                    # Set basic model parameters
                    config["model"]["encoder"] = encoder
                    config["model"]["decoder"] = decoder
                    config["model"]["dataset"] = dataset
                    config["model"]["seed"] = seed

                    # Set identifier
                    identifier = f"{dataset}_{encoder}_{decoder}_{seed}"
                    config["model"]["identifier"] = identifier

                    config["model"]["coord_descent"] = False

                    # Set forecast_length based on rollout
                    if "sindy_attention" in encoder or encoder in [
                        "moe_lstm",
                        "moe_gru",
                    ]:
                        if "_5" in encoder:
                            config["model"]["forecast_length"] = 5
                            config["model"]["encoder"] = encoder[:-2]
                        else:
                            config["model"]["forecast_length"] = 1
                    else:
                        config["model"]["forecast_length"] = 1

                    output_dir = top_dir / f"configs/test"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{identifier}.yaml"
                    with open(output_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    print(f"Generated config: {output_path}")


if __name__ == "__main__":
    main()
