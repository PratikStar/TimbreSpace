import csv
import os

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
# chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_6/checkpoints/last.ckpt")
chk_path = os.path.join(os.getcwd(), f"logw/logs/{config['model_params']['name']}/version_31/checkpoints/last.ckpt")

# chkpt = torch.load(chk_path, map_location=torch.device('cpu'))
model = MusicVAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                     map_location=torch.device('cpu'),
                                                     vae_model=vae_models[config['model_params']['name']](
                                                         **config['model_params']),
                                                     params=config)
## For generating embeddings
data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
batch, batch_di, key, offset = next(iter(dl))
# exit()
times = 1
for step, (batch, batch_di, key, offset) in enumerate(dl):
    batch = torch.squeeze(batch, 0)
    di_recons, _, _, _, z = model.forward(batch)

    di_recons = di_recons.detach().cpu().numpy()
    batch_di = torch.squeeze(batch_di, 0).cpu().numpy()
    batch = torch.squeeze(batch, 0).cpu().numpy()
    if step == times:
        break

    print(f"for file: {data.dataset.clips[key]}, Offset: {offset}")
    for i in range(di_recons.shape[0]):
        data.dataset.preprocessing_pipeline.visualizer.visualize_multiple([batch_di[i, 0, :, :], di_recons[i, 0, :, :]],
                                                                          ['original', 'recons'],
                                                                          "DI.wav", f"offset - {float(offset):0.2f} - {i}")

# a = torch.tensor(batch_di[0, 0, :, :])
# b = torch.tensor(di_recons[0, 0, :, :])
#
# l1 = F.l1_loss(a,b)
# l2 = F.mse_loss(a,b)
# mrstft = auraloss.freq.STFTLoss()
#
# ls = mrstft(a,b)