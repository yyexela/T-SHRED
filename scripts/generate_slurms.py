from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=16
#SBATCH --time=4-0
#SBATCH --nice=0

#SBATCH --job-name="{identifier}"
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/T-SHRED/logs/"{identifier}"_%j.out

#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexeyy@uw.edu

identifier={identifier}

repo="/mmfs1/home/alexeyy/storage/r4/T-SHRED"
datasets="/mmfs1/home/alexeyy/storage/data"

batch_size={batch_size}
dataset={dataset}
decoder={decoder}
decoder_depth={decoder_depth}
device={device}
dropout={dropout}
early_stop={early_stop}
encoder={encoder}
encoder_depth={encoder_depth}
epochs={epochs}
hidden_size={hidden_size}
lr={lr:0.2e}
n_heads={n_heads}
poly_order={poly_order}
save_every_n_epochs={save_every_n_epochs}
seed={seed}
window_length={window_length}
n_sensors={n_sensors}
n_well_tracks={n_well_tracks}
sindy_attention_weight={sindy_attention_weight}

echo "Running Apptainer"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer.sif --dataset "$dataset" --device "$device" --encoder "$encoder" --decoder "$decoder" --decoder_depth "$decoder_depth" --dropout "$dropout" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --hidden_size "$hidden_size" --lr "$lr" --n_heads "$n_heads" --poly_order "$poly_order" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --early_stop "$early_stop" --sindy_attention_weight "$sindy_attention_weight" --n_well_tracks "$n_well_tracks" --generate_test_plots --seed "$seed" --identifier "$identifier"

echo "Finished running Apptainer"\
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

datasets = ["planetswe", "sst", "plasma"]
encoders = ["gru", "sindy_loss_gru", "lstm", "sindy_loss_lstm", "vanilla_transformer", "sindy_loss_transformer", "sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]
decoders = ["mlp", "cnn"]
lrs = [1e-2, 1e-3]
encoder_depths = [1, 2, 3, 4]
poly_order = 1
device = "cuda:0"
batch_size = 128
dropout = 0.1
early_stop = 10
epochs = 100
save_every_n_epochs = 10
seeds = [0, 1, 2, 3, 4]
window_length = 50
decoder_depth = 1
sindy_attention_weight = 0.0
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
                            hidden_size = 100
                            n_heads = 4
                            memory = 32
                        elif dataset in ['planetswe', 'sst']:
                            n_sensors = 50
                            hidden_size = 100
                            n_heads = 4
                            memory = 64

                        identifier = f"{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}_s{seed}"

                        cmd = cmd_template.format(
                            dataset=dataset,
                            device=device,
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
                            memory=memory,
                            n_well_tracks=n_well_tracks,
                            seed=seed,
                            identifier=identifier,
                            partition=partition,
                        )

                        identifier = f'{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}_s{seed}'

                        total_count += 1

                        # Skip creating slurms that are completed
                        pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

                        if pickle_file.exists():
                            #print(f'Skipping {identifier}')
                            skip_count += 1
                            continue

                        with open(top_dir / 'slurms' / f'{identifier}.slurm', "w") as f:
                            f.write(cmd)
                            write_count += 1

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
