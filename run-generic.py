import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
import wandb
from datasets import TimbreDataModule
from experiment_timbre_transfer import TimbreTransferLM
from models import *
from utils import *
import hashlib, base64
from datetime import datetime
from random import random
import time
import os
import pathlib

print(f"\n\n\n==================== STARTING ===========================")
print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
print(f"Device: {device}")
print(os.environ)


### Config command line overrides
args, overrides = parse_args()
config = get_config(args.filename)
config = config_cmd_overrides(config, overrides)

if 'WANDB_SWEEP_ID' in os.environ:
    print(f"WANDB_SWEEP_ID: {os.environ['WANDB_SWEEP_ID']}")
    config.wandb.project = os.environ['WANDB_PROJECT']
    config.wandb.entity = os.environ['WANDB_ENTITY']

    # hasher = hashlib.sha1(str.encode(datetime.now().strftime("%Y%m%d%H%M%S")))
    # run_id = base64.urlsafe_b64encode(hasher.digest()).decode("utf-8")[:8]

    # we need this because the process from all the ranks would try to writ in the same wandb directory resulting in collisions/deadlocks
    wandb_dir = os.getcwd() + "/wandb/"
    if os.environ.get('LOCAL_RANK', None) is not None:
        wandb_dir += f"rank-{os.environ.get('LOCAL_RANK', None)}"

    p = pathlib.Path(wandb_dir)
    p.mkdir(parents=True, exist_ok=True)
    p.chmod(0o777)

    print(f"wandb_dir is {wandb_dir}")

    wandb.init(config=config.to_dict(),
               reinit=True,
               dir= wandb_dir
               # id=
               )

    print(wandb.config)
    config = Prodict.from_dict(merge(config, dict(wandb.config)))

if os.environ.get("LOCAL_RANK", None) is None:
    print("LOCAL_RANK is None")


data = TimbreDataModule(config.data_params, pin_memory=torch.cuda.device_count() != 0)
data.setup()
config = config_data_overrides(config, data)

config = config_device_overrides(config)

print(f"FINAL CONFIG: {config}")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

logger = get_logger(logger_type="wandb", config=config)
# model stuff
lightning_module = None
if 'load_path' in config['model_params']:
    print(f"Loading model from {config['model_params']['load_path']}")
    chk_path = os.path.join(os.getcwd(), config['model_params']['load_path'])
    lightning_module = TimbreTransferLM.load_from_checkpoint(checkpoint_path=chk_path,
                                                             map_location=torch.device(device))
else:
    model = vae_models[config['model_params']['name']](config['model_params'])
    lightning_module = TimbreTransferLM(model,
                                        config=config)

trainer = Trainer(logger=logger,
                  # check_val_every_n_epoch=5,
                  callbacks=[
                      LearningRateMonitor(),
                      ModelCheckpoint(dirpath=os.path.join(logger.save_dir, "checkpoints"),
                                      **config.exp_params.model_checkpoint),
                  ],
                  strategy=DDPPlugin(find_unused_parameters=False),
                  # strategy='ddp',
                  **config.trainer_params)

print(f"======= Training {config['model_params']['name']} =======")
trainer.fit(lightning_module, datamodule=data)
wandb.finish()
