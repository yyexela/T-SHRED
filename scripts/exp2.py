import os
import time
from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
time python -u {script_dir} --dataset {dataset} --device {device} --encoder {encoder} --decoder {decoder} --decoder_depth {decoder_depth} --dropout {dropout} --epochs {epochs} --save_every_n_epochs {save_every_n_epochs} --hidden_size {hidden_size} --lr {lr} --n_heads {n_heads} --poly_order {poly_order} --batch_size {batch_size} --encoder_depth {encoder_depth} --window_length {window_length} --early_stop {early_stop} --sindy_attention_weight {sindy_attention_weight} --n_well_tracks {n_well_tracks} --generate_test_plots --seed {seed} --identifier {identifier} --verbose 2>&1 | tee {log_path}
"""

# File paths
repo="/home/alexey/Research4/T-SHRED"
datasets="/home/alexey/Research4/T-SHRED/datasets"
script_dir = Path(repo) / 'scripts' / 'main.py'
log_dir = Path(repo) / 'logs'

# We will iterate through every combination of these
datasets = ["sst", "plasma", "planetswe"]
encoders = ["sindy_attention_transformer"]
decoders = ["mlp", "unet"]
lrs = [1e-3]
encoder_depths = [2]
poly_order = 1
device = "cuda:1"
batch_size = 128
dropout = 0.1
early_stop = 10
epochs = 200
save_every_n_epochs = 10
seeds = [0, 1, 2, 3, 4]
window_length = 50
decoder_depth = 1
sindy_attention_weight = 100.0
n_well_tracks = 10

skip_count = 0
write_count = 0
total_count = 0

for seed in seeds:
    for dataset in datasets:
        for encoder in encoders:
            for decoder in decoders:
                for lr in lrs:
                    for encoder_depth in encoder_depths:
                        if dataset == 'planetswe':
                            partition = 'ckpt-g2'
                        else:
                            partition = 'ckpt-g2'

                        if dataset == 'plasma':
                            n_sensors = 50
                            hidden_size = 6
                            n_heads = 2
                        elif dataset in ['planetswe', 'sst']:
                            n_sensors = 50
                            hidden_size = 6
                            n_heads = 2

                        identifier = f"{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}_s{seed}"

                        total_count += 1

                        # Skip creating slurms that are completed
                        pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

                        if pickle_file.exists():
                            print(f'Skipping {identifier}')
                            skip_count += 1
                            continue

                        log_path = log_dir / f'{identifier}.out'

                        cmd = cmd_template.format(
                            dataset=dataset,
                            device=device,
                            script_dir=script_dir,
                            batch_size=batch_size,
                            dropout=dropout,
                            encoder=encoder,
                            encoder_depth=encoder_depth,
                            decoder=decoder,
                            decoder_depth=decoder_depth,
                            early_stop=early_stop,
                            epochs=epochs,
                            n_heads=n_heads,
                            poly_order=poly_order,
                            save_every_n_epochs=save_every_n_epochs,
                            window_length=window_length,
                            lr=lr,
                            hidden_size=hidden_size,
                            n_sensors=n_sensors,
                            sindy_attention_weight=sindy_attention_weight,
                            log_path=log_path,
                            n_well_tracks=n_well_tracks,
                            seed=seed,
                            identifier=identifier,
                        )

                        print("-"*100)
                        print("Running", cmd)
                        print("-"*100)

                        os.system(cmd)
                        time.sleep(1.0)

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
