import os
from pathlib import Path

from experiment_music import MusicVAELightningModule
from experiment_timbre_transfer_flatten import TimbreTransferFlattenLM
from experiment_timbre_transfer import TimbreTransferLM
from models import *
from experiment import VAELightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from datasets import CelebAZipDataModule
from pytorch_lightning.plugins import DDPPlugin
from utils import *
from datasets import TimbreDataModule
from torchsummary import summary
import wandb

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")

device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
print(f"Device: {device}")

config = get_config(parse_args().filename)
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

## data stuff
data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
fb = next(iter(dl))
