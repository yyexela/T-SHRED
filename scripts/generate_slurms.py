from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

job_id={encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}_s{seed}

#SBATCH --account=amath
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=16
#SBATCH --time=4-0
#SBATCH --nice=0

#SBATCH --job-name="$job_id"
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/T-SHRED/logs/"$job_id"_%j.out

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
seed={seed}
window_length=50
n_sensors={n_sensors}
n_well_tracks=10
sindy_attention_weight=0.0

echo "Running Apptainer"

apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer.sif --dataset "$dataset" --device cuda:0 --encoder "$encoder" --decoder "$decoder" --decoder_depth "$decoder_depth" --device "$device" --dropout "$dropout" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --hidden_size "$hidden_size" --lr "$lr" --n_heads "$n_heads" --poly_order "$poly_order" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --early_stop "$early_stop" --sindy_attention_weight "$sindy_attention_weight" --n_well_tracks "$n_well_tracks" --generate_test_plots --seed "$seed" --job_id "$job_id"

echo "Finished running Apptainer"\
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

datasets = ["sst", "plasma"]
encoders = ["gru", "sindy_loss_gru", "lstm", "sindy_loss_lstm", "vanilla_transformer", "sindy_loss_transformer", "sindy_attention_transformer", "sindy_attention_sindy_loss_transformer"]
decoders = ["mlp", "unet"]
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

skip_count = 0
write_count = 0
total_count = 0

for seed in seeds:
    for dataset in datasets:
        for encoder in encoders:
            for decoder in decoders:
                for lr in lrs:
                    for encoder_depth in encoder_depths:
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
                            poly_order=poly_order,
                            seed=seed
                        )

                        identifier = f'{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}'

                        total_count += 1

                        # Skip creating slurms that are completed
                        pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

                        if pickle_file.exists():
                            print(f'Skipping {identifier}')
                            skip_count += 1
                            continue

                        with open(top_dir / 'slurms' / f'{identifier}.slurm', "w") as f:
                            f.write(cmd)
                            write_count += 1

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
