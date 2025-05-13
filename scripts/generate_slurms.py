from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-4
#SBATCH --nice=0

#SBATCH --job-name={encoder}_{decoder}_{dataset}_{lr:0.2e}
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer/logs/{encoder}_{decoder}_{dataset}_{lr:0.2e}_%j.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexeyy@uw.edu

repo="/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer"
datasets="/mmfs1/home/alexeyy/storage/data"

batch_size=128
dataset={dataset}
decoder={decoder}
decoder_depth=2
device=cuda:0
dropout=0.1
encoder={encoder}
encoder_depth=2
epochs=500
hidden_size={hidden_size}
lr={lr:0.2e}
n_heads=2
poly_order=2
save_every_n_epochs=10
window_length=50
n_sensors={n_sensors}


echo "Running Apptainer"

apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer.sif --dataset "$dataset" --device cuda:0 --encoder "$encoder" --decoder "$decoder" --decoder_depth "$decoder_depth" --device "$device" --dropout "$dropout" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --hidden_size "$hidden_size" --lr "$lr" --n_heads "$n_heads" --poly_order "$poly_order" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --verbose

echo "Finished running Apptainer"\
"""

datasets = ["sst", "plasma", "active_matter", "planetswe", "active_matter_pod", "planetswe_pod"]
encoders = ["lstm", "vanilla_transformer", "sindy_attention_transformer"]
decoders = ["mlp", "unet"]
lrs = [1e-2, 1e-3, 1e-4, 1e-5]

for dataset in datasets:
    for encoder in encoders:
        for decoder in decoders:
            for lr in lrs:
                if dataset == 'plasma':
                    n_sensors = 5
                    hidden_size = 4
                elif dataset in ['active_matter_pod', 'planetswe_pod']:
                    n_sensors = 15
                    hidden_size = 8
                else:
                    n_sensors = 50
                    hidden_size = 8

                if dataset in ['active_matter', 'planetswe']:
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
                    memory=memory
                )

                with open(top_dir / 'slurms' / f'{encoder}_{decoder}_{dataset}_{lr:0.2e}.slurm', "w") as f:
                    f.write(cmd)




