from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=16
#SBATCH --time=4-0
#SBATCH --nice=0

#SBATCH --job-name={encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/T-SHRED/logs/{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}_%j.out

#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexeyy@uw.edu

repo="/mmfs1/home/alexeyy/storage/r4/T-SHRED"
datasets="/mmfs1/home/alexeyy/storage/data"

batch_size=128
dataset={dataset}
decoder={decoder}
decoder_depth={decoder_depth}
device=cuda:0
dropout=0.1
early_stop=20
encoder={encoder}
encoder_depth={encoder_depth}
epochs=100
hidden_size={hidden_size}
lr={lr:0.2e}
n_heads=2
poly_order={poly_order}
save_every_n_epochs=10
window_length=50
n_sensors={n_sensors}
n_well_tracks=10

echo "Running Apptainer"

apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer.sif --dataset "$dataset" --device cuda:0 --encoder "$encoder" --decoder "$decoder" --decoder_depth "$decoder_depth" --device "$device" --dropout "$dropout" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --hidden_size "$hidden_size" --lr "$lr" --n_heads "$n_heads" --poly_order "$poly_order" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --early_stop "$early_stop" --n_well_tracks "$n_well_tracks" --verbose

echo "Finished running Apptainer"\
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

# We will iterate through every combination of these
datasets = ["planetswe_full", "sst", "plasma"]
encoders = ["lstm", "gru", "sindy_loss_lstm", "sindy_loss_gru", "vanilla_transformer", "sindy_loss_transformer", "sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]
decoders = ["mlp", "unet"]
lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
poly_orders = [1]

# These two will be zipped pairwise
encoder_depths = [1, 2, 3, 4]
decoder_depths = [1, 1, 1, 1]

skip_count = 0
write_count = 0
total_count = 0

for dataset in datasets:
    for encoder in encoders:
        for decoder in decoders:
            for lr in lrs:
                for encoder_depth, decoder_depth in zip(encoder_depths, decoder_depths):
                    for poly_order in poly_orders:
                        if dataset == 'plasma':
                            n_sensors = 5
                            hidden_size = 4
                        elif dataset in ['planetswe_full', 'planetswe_pod']:
                            if poly_order == 2:
                                continue

                            n_sensors = 50
                            hidden_size = 100
                        else: # sst
                            n_sensors = 50
                            hidden_size = 100

                        if dataset in ['planetswe_full', 'planetswe_pod']:
                            memory = 64
                        else:
                            memory = 32

                        cmd = cmd_template.format(
                            dataset=dataset,
                            encoder=encoder,
                            decoder=decoder,
                            lr=lr,
                            hidden_size=hidden_size,
                            n_sensors=n_sensors,
                            memory=memory,
                            encoder_depth=encoder_depth,
                            decoder_depth=decoder_depth,
                            poly_order=poly_order
                        )

                        identifier = f'{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}'

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
