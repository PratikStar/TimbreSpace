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


### Config command line overrides
args, overrides = parse_args()
config = get_config(args.filename)
config = config_cmd_overrides(config, overrides)

data = TimbreDataModule(config.data_params, pin_memory=torch.cuda.device_count() != 0)
data.setup()

