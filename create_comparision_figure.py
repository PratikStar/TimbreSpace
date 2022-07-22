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

chk_paths = [
    os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_0/checkpoints/last.ckpt"),  # 16
    # os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_2/checkpoints/last.ckpt"), # 32
    # os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_4/checkpoints/last.ckpt"), # 64
    # os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_9/checkpoints/last.ckpt"), # 64
    # os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_10/checkpoints/last.ckpt"), # 32
    os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_11/checkpoints/last.ckpt")  # 16
]

data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)  # Update the config
data.setup()
dl = data.train_dataloader()
batch, batch_di, key, offset = next(iter(dl))
batch = torch.squeeze(batch, 0)
batch_di = torch.squeeze(batch_di, 0).cpu().numpy()
batch_vis = batch.cpu().numpy()
spectrograms = [batch_vis[:, 0, :, :], batch_di[:, 0, :, :]]

for chk_path in chk_paths:
    model = MusicVAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                         map_location=torch.device('cpu'),
                                                         )

    di_recons, _, _, _, z = model.forward(batch)
    di_recons = di_recons.detach().cpu().numpy()
    spectrograms.append(di_recons[:, 0, :, :])

data.dataset.preprocessing_pipeline.visualizer.visualize_multiple(
    spectrograms,
    file_dir=Path("./out"),
    col_titles=["Reamped", "Expected DI", "Recons (L2 loss)", "Recons (L1 loss)"],
    filename="reconstruction-256x16.png",
    title=f"Reconstructions with n_frames= {16} ({0.18} seconds)")
