"""
This script generates configuration files for hyperparameter tuning the models.
Writes configuration files into the `configs/**/tuning_config/` directories.
These configurations are used by `generate_tuning_bash.py` to generate bash scripts for hyperparameter tuning.

If the configuration file has already generated a `optimal_params_*.yaml` file in the `results/` directory, it is skipped.
"""

import copy
import yaml
import shutil
from pathlib import Path

encoder_dict = {
    "gru": "GRU",
    "lstm": "LSTM",
    "moe_gru": "MOE-GRU",
    "moe_lstm": "MOE-LSTM",
    "sindy_loss_gru": "SL-GRU",
    "sindy_loss_lstm": "SL-LSTM",
    "vanilla_transformer": "T",
    "sindy_loss_transformer": "SLT",
    "sindy_attention_transformer": "SA-T",
    "sindy_attention_sindy_loss_transformer": "SASL-T",
    "sindy_attention_transformer_5": "SA-T-5",
    "sindy_attention_sindy_loss_transformer_5": "SASL-T-5",
}

decoder_dict = {
    "mlp": "MLP",
    "cnn": "CNN",
}

n_seeds = 10
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
datasets = ["sst", "plasma", "planetswe"]

top_dir = Path(__file__).parent.parent


def main():
    # Create output directory and save config, clean up old configs
    configs_dir = top_dir / "configs"
    encoder_dirs = [
        d.name for d in configs_dir.iterdir() if d.is_dir() and d.name != "template"
    ]
    for encoder_dir in encoder_dirs:
        output_dir = top_dir / f"configs/{encoder_dir}/tuning_config"
        shutil.rmtree(output_dir, ignore_errors=True)

    # Load template config
    template_path = top_dir / "configs/template/template.yaml"
    with open(template_path, "r") as f:
        template_config = yaml.safe_load(f)

    config_count = 0
    skip_count = 0

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

                    # Check if config has been optimized
                    results_path = (
                        Path("/") / "home" / "alexey" / "Git" / "T-SHRED" / "results"
                    )
                    results_path = (
                        results_path / identifier / f"optimal_params_{dataset}.yaml"
                    )
                    if results_path.exists():
                        skip_count += 1
                        continue

                    # Set hyperparameters to remove to empty
                    hyperparams_to_remove = []

                    config["model"]["coord_descent"] = False

                    hyperparams_to_remove.append("coord_descent_model_n_epochs")
                    hyperparams_to_remove.append("coord_descent_model_lr")
                    hyperparams_to_remove.append("coord_descent_sindy_layer_n_epochs")
                    hyperparams_to_remove.append("coord_descent_sindy_layer_lr")

                    # SINDy Loss options
                    if "sindy_loss" not in encoder:
                        hyperparams_to_remove.append("sindy_loss_weight")
                    if "sindy_attention" not in encoder:
                        hyperparams_to_remove.append("sindy_layer_weight")

                    # Set forecast_length based on rollout
                    if "sindy_attention" in encoder:
                        if "_5" in encoder:
                            config["model"]["forecast_length"] = 5
                            config["model"]["encoder"] = encoder[:-2]
                        else:
                            config["model"]["forecast_length"] = 1
                    else:
                        config["model"]["forecast_length"] = 1

                    # Remove relevant hyperparameters
                    for param in hyperparams_to_remove:
                        if param in config["hyperparameters"]:
                            del config["hyperparameters"][param]

                    output_dir = (
                        top_dir / f"configs/{encoder_dict[encoder]}/tuning_config"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{identifier}.yaml"
                    with open(output_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    print(f"Generated config: {output_path}")

                    config_count += 1

    print(f"Total configs generated: {config_count}")
    print(f"Total configs skipped: {skip_count}")


if __name__ == "__main__":
    main()
