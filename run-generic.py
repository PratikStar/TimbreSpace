import os

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
import wandb
from datasets import TimbreDataModule
from experiment_timbre_transfer import TimbreTransferLM
from models import *
from utils import *

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
print(f"Device: {device}")

# print(os.environ.keys())

### Config command line overrides
args, overrides = parse_args()
config = get_config(args.filename)
config = Prodict.from_dict(merge(config, overrides))

# Device dependant config
if device == "cpu":
    config.trainer_params.gpus = []
###

if 'WANDB_SWEEP_ID' in os.environ:
    print(f"WANDB_SWEEP_ID: {os.environ['WANDB_SWEEP_ID']}")
    config.wandb.project = os.environ['WANDB_PROJECT']
    config.wandb.entity = os.environ['WANDB_ENTITY']
    wandb.init()
    # print(wandb.config)
    config = Prodict.from_dict(merge(config, dict(wandb.config)))
else:
    wandb.init(project=config.wandb.project, entity=config.wandb.entity)
    wandb.config.update(dict(config))
# print(f"CONFIG: {config}")

print(wandb.run.sweep_id)
print(wandb.run.id)

## data stuff
data = TimbreDataModule(config.data_params, pin_memory=torch.cuda.device_count() != 0)
data.setup()
dl = data.train_dataloader()
fb = next(iter(dl))
# print("DATA STUFF DONE")

# config update for dependant params
config.model_params.timbre_encoder.spectrogram_dims[1] = fb[0].shape[-2]
config.model_params.timbre_encoder.spectrogram_dims[2] = fb[0].shape[-1]
config.model_params.decoder.di_spectrogram_dims[1] = fb[0].shape[-2]
config.model_params.decoder.di_spectrogram_dims[2] = fb[0].shape[-1]
# print("model params set")
# print(f"CONFIG: {config}")


if config.model_params.merge_encoding == "sandwich":
    config.model_params.timbre_encoder.latent_dim = fb[0].shape[-2]  # this depends upon merge method
elif config.model_params.merge_encoding == "condconv":
    print("merge encoding condconv")

print(f"FINAL CONFIG: {config}")
# print(f"{{**dict(config.exp_params.model_checkpoint)}}")
# exit()

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )
print("SEED EVERYTHING")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

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


# exit()
def get_logger_path(logger="wandb"):
    path = ""
    if logger=="wandb":
        if wandb.run.sweep_id is not None:
            path += "sweep-" + str(wandb.run.sweep_id) + "/"
        if wandb.run.id is not None:
            path += "run-" + str(wandb.run.id) + "/"
    return path

wandb_logger = WandbLogger(project=config.wandb.project,
                           save_dir=f"{config.logging_params.save_dir}{get_logger_path('wandb')}")

trainer = Trainer(logger=wandb_logger,
                  # check_val_every_n_epoch=5,
                  callbacks=[
                      LearningRateMonitor(),
                      ModelCheckpoint(dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      **config.exp_params.model_checkpoint),
                  ],
                  # strategy=DDPPlugin(find_unused_parameters=False),
                  strategy='ddp',
                  gpus=-1,

                  **config.trainer_params)
print(f"======= Training {config['model_params']['name']} =======")
# exit()
trainer.fit(lightning_module, datamodule=data)


