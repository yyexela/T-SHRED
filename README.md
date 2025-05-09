## Transformer SINDy-SHRED

Transformer SINDy-SHRED pacakge

## Environment

The environment is described in `pyproject.toml`. To install, please run:

```
$ pyenv install 3.13
$ pyenv local 3.13
$ python -m venv venv
$ source venv/bin/activate
$ pip install .
```

## File Structure

```
.
├── apptainer
│   └── apptainer.def
├── checkpoints
├── datasets
│   ├── plasma
│   ├── sst
│   └── the_well
│       ├ active_matter
│       └ planetswe
├── figures
├── logs
├── notebooks
│   ├── ROM_plasma.ipynb
│   └── the_well.ipynb
├── pyproject.toml
├── README.md
├── scripts
│   └── main.py
├── slurms
│   └── sst.slurm
└── src
    ├── datasets.py
    ├── forecasts.py
    ├── helpers.py
    ├── __init__.py
    ├── models.py
    ├── processdata.py
    └── transformer_model.py
```

- `apptainer`:
  - Contains `apptainer.def` for creating a container on hyak.
- `checkpoints`: 
  - Contains the saved models during running. Each run, we save the model with the best "validation" score as well as the "latest" model as determined by the number of epochs run.
- `datasets`: 
  - Please download the datsets into the respective folders, this is the expected structure.
- `figures`: 
  - Folder to store figures.
- `logs`: 
  - Folder to store logs.
- `notebooks`: 
  - Helpful notebooks. `ROM_plasma.ipynb` is David's example of running Plasma data. `the_well.ipynb` is how to parse The Well's data.
- `pyproject.tml`: 
  - Python environment file.
- `README.md`: 
  - This file
- `scripts`: 
  - Contains the primary entry-point `main.py` to training and evaluating a model.
- `slurms`: 
  - Contains hyak slurm files for batch runs.
- `src`: 
  - Contains primary package functions/code.

## Running in CLI

To run in the command line, make sure you set up the environment as described above. Then, you can execute a command like so:

```
$(venv) time python -u scripts/main.py --dataset sst --encoder transformer --decoder mlp --epochs 100 --save_every_n_epochs 10 --batch_size 5 --encoder_depth 1 --window_length 10 --dim_feedforward 128 --verbose 2>&1 | tee logs/active_matter.txt
```

## Running on Hyak

Hyak is confusing, I made a [blog post](https://yyexela.github.io/blog/blog/research/using-hyak-guide#step-4-running-an-interactive-session) about it a while back which can be used as a reference for the below commands.

To run on Hyak, you first need to build a container. Do so in an interactive session.

```
salloc -A amath -p gpu-rtx6k -N 1 -n 2 -c 4 -G 1 --mem=4G --time=5:00:00 --nice=0
```

Once you're in the interactive session, you can build the apptainer with the following command:

```
$ cd apptainer
$ apptainer build ./apptainer.sif ./apptainer.def
```

Then, you need to create a `.slurm` file in the `slurms` directory. You can use this as a template:

```
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
#SBATCH --mail-user=<NETID>@uw.edu

repo="<REPO>"
datasets="<DATA>"

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
```

You'll have to swap out `repo` and `datasets` with the location you have your `repo` and `datasets` downloaded. Note that these both should be stored on a `/gscratch` folder so that it doesn't take up the `10G` maximum storage in home directories. Also, you'll have to download/upload the datasets to your `datasets` folder.

Also, change the above email to your UW email.

Then, you can run the code with `sbatch <filename>.slurm`!
