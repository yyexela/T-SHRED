import os
import sys
import copy
import yaml
import shutil
from pathlib import Path

encoder_dict = {
    "gru": "GRU",
    "lstm": "LSTM",
    "sindy_loss_gru": "SL-GRU",
    "sindy_loss_lstm": "SL-LSTM",
    "vanilla_transformer": "T",
    "sindy_loss_transformer": "SLT",
    "sindy_attention_transformer": "SA-T",
    "sindy_attention_sindy_loss_transformer": "SAR-T",
    "sindy_attention_transformer_5": "SA-T-5",
    "sindy_attention_sindy_loss_transformer_5": "SAR-T-5",
}

n_seeds = 5
encoders = ["sindy_attention_transformer_5", "sindy_attention_sindy_loss_transformer_5"] # "gru", "lstm", "sindy_loss_gru", "sindy_loss_lstm", "vanilla_transformer", "sindy_loss_transformer", "sindy_attention_transformer", "sindy_attention_sindy_loss_transformer", "sindy_attention_transformer_5", "sindy_attention_sindy_loss_transformer_5"
decoders = ["mlp", "cnn"]
datasets = ["sst"] # , "plasma", "planetswe"

top_dir = Path(__file__).parent.parent

def main():
    # Create output directory and save config
    for encoder in encoders:
        output_dir = top_dir / f"configs/{encoder_dict[encoder]}/tuning_config"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load template config
    template_path = top_dir / "configs/template/template_planetswe.yaml"
    with open(template_path, 'r') as f:
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
                    config['model']['encoder'] = encoder
                    config['model']['decoder'] = decoder
                    config['model']['dataset'] = dataset
                    config['model']['seed'] = seed
                    
                    # Set identifier
                    identifier = f"{dataset}_{encoder}_{decoder}_{seed}"
                    config['model']['identifier'] = identifier

                    # Check if config has been optimized
                    results_path = Path("/") / "home" / "alexey" / "Git" / "T-SHRED" / "results"
                    results_path = results_path / identifier / f"optimal_params_{dataset}.yaml"
                    if results_path.exists():
                        skip_count += 1
                        continue

                    # Set hyperparameters to remove to empty
                    hyperparams_to_remove = []

                    # Set coord_descent based on sindy_attention
                    if "sindy_attention" in encoder:
                        config['model']['coord_descent'] = True
                        
                        hyperparams_to_remove.append("lr")
                    else:
                        config['model']['coord_descent'] = False

                        hyperparams_to_remove.append("coord_descent_model_n_epochs")
                        hyperparams_to_remove.append("coord_descent_model_lr")
                        hyperparams_to_remove.append("coord_descent_sindy_attention_n_epochs")
                        hyperparams_to_remove.append("coord_descent_sindy_attention_lr")

                    # SINDy Loss options
                    if "sindy_loss" not in encoder:
                        hyperparams_to_remove.append("sindy_loss_weight")
                    
                    # Set forecast_length based on rollout
                    if "sindy_attention" in encoder:
                        if "_5" in encoder:
                            config['model']['forecast_length'] = 5
                            config['model']['encoder'] = encoder[:-2]
                        else:
                            config['model']['forecast_length'] = 1

                        # Only 1 encoder depth supported
                        hyperparams_to_remove.append("encoder_depth")
                        config['model']['encoder_depth'] = 1
                    else:
                        config['model']['forecast_length'] = 1
 
                    # Remove relevant hyperparameters
                    for param in hyperparams_to_remove:
                        if param in config['hyperparameters']:
                            del config['hyperparameters'][param]
                     
                    output_dir = top_dir / f"configs/{encoder_dict[encoder]}/tuning_config"
                    output_path = output_dir / f"{identifier}.yaml"
                    with open(output_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    print(f"Generated config: {output_path}")
                    
                    config_count += 1
    
    print(f"Total configs generated: {config_count}")
    print(f"Total configs skipped: {skip_count}")

if __name__ == "__main__":
    main()
