import os
import time
from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
time python -u {script_dir} --dataset {dataset} --device {device} --encoder {encoder} --decoder {decoder} --decoder_depth {decoder_depth} --device {device} --dropout {dropout} --epochs {epochs} --save_every_n_epochs {save_every_n_epochs} --hidden_size {hidden_size} --lr {lr} --n_heads {n_heads} --poly_order {poly_order} --batch_size {batch_size} --encoder_depth {encoder_depth} --window_length {window_length} --early_stop {early_stop} --skip_load_checkpoint --verbose 2>&1 | tee {log_path}
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

# Repo directories
repo="/home/alexey/Research4/T-SHRED"
datasets="/home/alexey/Research4/T-SHRED/datasets"
script_dir = Path(repo) / 'scripts' / 'main.py'
log_dir = Path(repo) / 'logs'

# Fixed parameters
batch_size=128
device="cuda:0"
dropout=0.1
early_stop=20
epochs=100
n_heads=2
poly_order=2
save_every_n_epochs=10
window_length=50

# We will iterate through every combination of these
datasets = ["sst", "plasma", "planetswe_pod", "gray_scott_reaction_diffusion_pod"]
encoders = ["sindy_loss_lstm"]
decoders = ["mlp", "unet"]
lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# These two will be zipped pairwise
encoder_depths = [1, 2, 3, 3, 4, 4]
decoder_depths = [1, 2, 1, 2, 1, 2]

for dataset in datasets:
    for encoder in encoders:
        for decoder in decoders:
            for lr in lrs:
                for encoder_depth, decoder_depth in zip(encoder_depths, decoder_depths):
                    if dataset == 'plasma':
                        n_sensors = 5
                        hidden_size = 4
                    elif dataset in ['gray_scott_reaction_diffusion_pod', 'planetswe_pod']:
                        n_sensors = 50
                        hidden_size = 100
                    else: # sst
                        n_sensors = 50
                        hidden_size = 100

                    if dataset in ['gray_scott_reaction_diffusion_pod', 'planetswe_pod']:
                        memory = 64
                    else:
                        memory = 32

                    identifier = f'{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}'
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
                        memory=memory,
                        log_path=log_path 
                    )

                    print("-"*100)
                    print("Running", cmd)
                    print("-"*100)

                    os.system(cmd)
                    time.sleep(1.0)
