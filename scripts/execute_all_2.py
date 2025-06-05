import os
import time
from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
time python -u {script_dir} --dataset {dataset} --device {device} --encoder {encoder} --decoder {decoder} --decoder_depth {decoder_depth} --device {device} --dropout {dropout} --epochs {epochs} --save_every_n_epochs {save_every_n_epochs} --hidden_size {hidden_size} --lr {lr} --n_heads {n_heads} --poly_order {poly_order} --batch_size {batch_size} --encoder_depth {encoder_depth} --window_length {window_length} --early_stop {early_stop} --sindy_attention_weight {sindy_attention_weight} --generate_test_plots 2>&1 | tee {log_path}
"""

# File paths
repo="/home/alexey/Research4/T-SHRED"
datasets="/home/alexey/Research4/T-SHRED/datasets"
script_dir = Path(repo) / 'scripts' / 'main.py'
log_dir = Path(repo) / 'logs'

# We will iterate through every combination of these
datasets = ["planetswe"]
encoders = ["vanilla_transformer", "sindy_loss_transformer"]
decoders = ["mlp", "unet"]
lrs = [1e-2, 1e-3]
encoder_depths = [1, 2, 3, 4]
poly_order = 1
device = "cuda:1"
batch_size = 128
dropout = 0.1
early_stop = 10
epochs = 100
save_every_n_epochs = 10
window_length = 50
decoder_depth = 1
sindy_attention_weight = 0.0

skip_count = 0
write_count = 0
total_count = 0

for dataset in datasets:
    for encoder in encoders:
        for decoder in decoders:
            for lr in lrs:
                for encoder_depth in encoder_depths:
                    if dataset == 'plasma':
                        n_sensors = 3
                        hidden_size = 6
                        n_heads = 2
                        memory = 32
                    elif dataset in ['planetswe', 'sst']:
                        n_sensors = 50
                        hidden_size = 100
                        n_heads = 4
                        memory = 64

                    identifier = f'{encoder}_{decoder}_{dataset}_e{encoder_depth}_d{decoder_depth}_lr{lr:0.2e}_p{poly_order}'

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
                        memory=memory,
                        log_path=log_path,
                    )

                    print("-"*100)
                    print("Running", cmd)
                    print("-"*100)

                    os.system(cmd)
                    time.sleep(1.0)

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
