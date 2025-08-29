import os
import sys
import yaml
import shutil
import copy
from pathlib import Path

encoder_dict = {
    "gru": "GRU",
    "lstm": "LSTM",
    "sindy_loss_gru": "SL-GRU",
    "sindy_loss_lstm": "SL-LSTM",
    "vanilla_transformer": "T",
    "sindy_attention_transformer": "SA-T",
    "sindy_loss_transformer": "SLT",
    "sindy_attention_sindy_loss_transformer": "SASLT-T",
    "sindy_attention_transformer_rollout": "SAR-T",
    "sindy_attention_sindy_loss_transformer_rollout": "SASLR-T",
}
n_seeds = 20
encoders = ["gru", "lstm", "sindy_loss_gru", "sindy_loss_lstm", "vanilla_transformer", "sindy_attention_transformer", "sindy_loss_transformer", "sindy_attention_sindy_loss_transformer", "sindy_attention_transformer_rollout", "sindy_attention_sindy_loss_transformer_rollout"]
decoders = ["mlp", "cnn"]
datasets = ["sst", "planetswe", "plasma"]

def main():
    # Load template config
    template_path = Path("configs/template/template.yaml")
    with open(template_path, 'r') as f:
        template_config = yaml.safe_load(f)
    
    config_count = 0
    
    # Iterate over seeds
    for seed in range(n_seeds):
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
                    if "rollout" in encoder:
                        config['model']['forecast_length'] = 5

                        # Only 1 encoder depth supported
                        hyperparams_to_remove.append("encoder_depth")
                        config['model']['encoder_depth'] = 1
                    else:
                        config['model']['forecast_length'] = 1
                    
                    # Remove relevant hyperparameters
                    for param in hyperparams_to_remove:
                        if param in config['hyperparameters']:
                            del config['hyperparameters'][param]
                    
                    # Create output directory and save config
                    output_dir = Path(f"configs/{encoder_dict[encoder]}/tuning_config")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{identifier}.yaml"
                    with open(output_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    
                    print(f"Generated config: {output_path}")
                    config_count += 1
    
    print(f"\nTotal configs generated: {config_count}")

if __name__ == "__main__":
    main()
