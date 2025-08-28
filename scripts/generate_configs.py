import os
import sys
import yaml
import shutil
from pathlib import Path

# The code does the following:
# 1. Loads T-SHRED/configs/template/template.yaml
# 2. Iterates over seeds (provided n_seeds) from 0 to n_seeds-1
# 3. Iteratres over all encoder types
#   "gru", "lstm", "sindy_loss_gru", "sindy_loss_lstm", "vanilla_transformer", "sindy_attention_transformer", "sindy_loss_transformer", "sindy_attention_sindy_loss_transformer", "sindy_attention_transformer_rollout", or "sindy_attention_sindy_loss_transformer_rollout"
# 4. Iterates over all decoder types
#   "mlp", "cnn"
# 5. Iterates over all datasets
#   "sst", "planetswe", "plasma"
# 6. Removes the following if an encoder does not contain "sindy_attention": "sindy_attention_weight", "sindy_attention_loss", "coord_descent_model_n_epochs", "coord_descent_sindy_attention_n_epochs", "coord_descent_model_lr", "coord_descent_sindy_attention_lr"
# 7. Sets "forecast_length" to 5 and "coord_descent" to true if encoder contains "rollout" and 1 and false otherwise
# 8. Sets "identifier" to f"{dataset}_{encoder}_{decoder}_{seed}"
# 9. Saves the config file to T-SHRED/configs/{encoder_dict[encoder]}/tuning_config/{identifier}.yaml

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
n_seeds = 1
encoders = ["gru", "lstm", "sindy_loss_gru", "sindy_loss_lstm", "vanilla_transformer", "sindy_attention_transformer", "sindy_loss_transformer", "sindy_attention_sindy_loss_transformer", "sindy_attention_transformer_rollout", "sindy_attention_sindy_loss_transformer_rollout"]
decoders = ["mlp", "cnn"]
datasets = ["sst", "planetswe", "plasma"]

def main():
    # 1. Load template config
    template_path = Path("configs/template/template.yaml")
    with open(template_path, 'r') as f:
        template_config = yaml.safe_load(f)
    
    config_count = 0
    
    # 2. Iterate over seeds
    for seed in range(n_seeds):
        # 3. Iterate over encoder types
        for encoder in encoders:
            # 4. Iterate over decoder types
            for decoder in decoders:
                # 5. Iterate over datasets
                for dataset in datasets:
                    # Create a copy of the template config
                    config = template_config.copy()
                    
                    # Set basic model parameters
                    config['model']['encoder'] = encoder
                    config['model']['decoder'] = decoder
                    config['model']['dataset'] = dataset
                    config['model']['seed'] = seed
                    
                    # 8. Set identifier
                    identifier = f"{dataset}_{encoder}_{decoder}_{seed}"
                    config['model']['identifier'] = identifier
                    
                    # 7. Set forecast_length and coord_descent based on rollout
                    if "rollout" in encoder:
                        config['model']['forecast_length'] = 5
                        config['model']['coord_descent'] = True
                    else:
                        config['model']['forecast_length'] = 1
                        config['model']['coord_descent'] = False
                    
                    # 6. Remove sindy_attention related parameters if encoder doesn't contain "sindy_attention"
                    if "sindy_attention" not in encoder:
                        hyperparams_to_remove = [
                            "sindy_attention_weight",
                            "coord_descent_model_n_epochs",
                            "coord_descent_sindy_attention_n_epochs",
                            "coord_descent_model_lr",
                            "coord_descent_sindy_attention_lr"
                        ]
                    else:
                        hyperparams_to_remove = [
                            "lr"
                        ]
                    for param in hyperparams_to_remove:
                        if param in config['hyperparameters']:
                            del config['hyperparameters'][param]
                    
                    # 9. Create output directory and save config
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
