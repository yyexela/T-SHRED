#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0
#SBATCH --nice=0

#SBATCH --job-name=vanilla_transformer_unet_active_matter_1.00e-03
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer/logs/vanilla_transformer_unet_active_matter_1.00e-03_%j.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexeyy@uw.edu

repo="/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer"
datasets="/mmfs1/home/alexeyy/storage/data"

batch_size=128
dataset=active_matter
decoder=unet
decoder_depth=1
device=cuda:0
dropout=0.1
encoder=vanilla_transformer
encoder_depth=1
epochs=500
hidden_size=8
lr=1.00e-03
n_heads=2
poly_order=2
save_every_n_epochs=10
window_length=50
n_sensors=50


echo "Running Apptainer"

apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer_distributed.sif --dataset "$dataset" --device cuda:0 --encoder "$encoder" --decoder "$decoder" --decoder_depth "$decoder_depth" --device "$device" --dropout "$dropout" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --hidden_size "$hidden_size" --lr "$lr" --n_heads "$n_heads" --poly_order "$poly_order" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --verbose

echo "Finished running Apptainer"