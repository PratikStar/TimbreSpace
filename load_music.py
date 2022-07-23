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
from experiment_music import MusicVAELightningModule
from models import *
from utils.utils import *
import numpy as np

## Load model
config = get_config(os.path.join(os.getcwd(), 'configs/music_vae.yaml'))
chk_path = os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_11/checkpoints/last.ckpt")

# chkpt = torch.load(chk_path, map_location=torch.device('cpu'))
model = MusicVAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                     map_location=torch.device('cpu'),
                                                     )
# exit()
## For generating embeddings
data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
batch, batch_di, key, offset = next(iter(dl))
exit()
times = 1
for step, (batch, batch_di, key, offset) in enumerate(dl):
    batch = torch.squeeze(batch, 0)
    di_recons, _, _, _, z = model.forward(batch)

    di_recons = di_recons.detach().cpu().numpy()
    batch_di = torch.squeeze(batch_di, 0).cpu().numpy()
    batch = batch.cpu().numpy()
    clip_name = data.dataset.clips[key]

    if step == times:
        break
    data.dataset.preprocessing_pipeline.visualizer.visualize_multiple(
        [batch[:, 0, :, :], batch_di[:, 0, :, :], di_recons[:, 0, :, :]],
        file_dir=Path(chk_path).parents[1] / 'recons',
        col_titles=["Reamped", "Expected DI", "Reconstruction DI"],
        title=f"Name: {model.config_dump['model_params']['name']}, Epochs: {model.config_dump['trainer_params']['max_epochs']}, Samples: {model.config_dump['model_params']['spectrogram_dims'][-1]}, Loss: {model.config_dump['model_params']['loss']['function']}")

# a = torch.tensor(batch_di[0, 0, :, :])
# b = torch.tensor(di_recons[0, 0, :, :])
#
# l1 = F.l1_loss(a,b)
# l2 = F.mse_loss(a,b)
# mrstft = auraloss.freq.STFTLoss()
#
# ls = mrstft(a,b)