import csv
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
import auraloss

from datasets import TimbreDataModule
from datasets.audio_stft import AudioSTFT, AudioSTFTDataModule
from experiment_music import MusicVAELightningModule
from models import *
from utils.utils import *
import numpy as np

## Load model
config = get_config(os.path.join(os.getcwd(), 'configs/vocoder.yaml'))
print(config.data_params)

data = AudioSTFTDataModule(config.data_params, pin_memory=False)

data.setup()
dl = data.train_dataloader()
mel, audio = next(iter(dl))
print(mel.shape)
# exit()
# ls = mrstft(a,b)