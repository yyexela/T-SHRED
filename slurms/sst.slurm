#!/bin/bash
  
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --nice=0

#SBATCH --job-name=train_sst
#SBATCH --output=/mmfs1/home/alexeyy/storage/r4/SINDy-Transformer/logs/output.out

#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexeyy@uw.edu

cd /home/alexeyy/storage/r4/SINDy-Transformer

