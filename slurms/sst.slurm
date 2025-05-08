#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0
#SBATCH --nice=0

#SBATCH --job-name=sst
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer/logs/sst_%j.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexeyy@uw.edu

repo="/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer"
datasets="/mmfs1/home/alexeyy/storage/data"

dataset=sst
encoder=transformer
decoder=mlp
epochs=100
save_every_n_epochs=10
batch_size=5
window_length=10
dim_feedforward=128
encoder_depth=1

echo "Running Apptainer"
apptainer run --nv --bind "$repo":/app/code --bind "$datasets":'/app/code/datasets' "$repo"/apptainer/apptainer.sif --dataset "$dataset" --device cuda:0 --encoder "$encoder" --decoder "$decoder" --epochs "$epochs" --save_every_n_epochs "$save_every_n_epochs" --batch_size "$batch_size" --encoder_depth "$encoder_depth" --window_length "$window_length" --dim_feedforward "$dim_feedforward" --verbose
echo "Finished running Apptainer"

