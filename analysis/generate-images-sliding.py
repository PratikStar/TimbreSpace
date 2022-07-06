import csv
import os

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.celeba import CelebAZipDatasetWithFilter, CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from utils.utils import *
import numpy as np

## Load model
config = get_config(os.path.join(os.getcwd(), 'configs/vae.yaml'))
chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
# chkpt = torch.load(chk_path, map_location=torch.device('cpu'))
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model=vae_models[config['model_params']['name']](
                                                    **config['model_params']),
                                                params=config['exp_params'])

## For saving slided images
z = torch.randn(128)
s = model.model.decode(z)

vutils.save_image(s, f'../sliding/sample-OG.png', normalize=True)
for dim in range(0, 128):
    arr = []
    for i in np.arange(-10, 10, 1):
        z[dim] = i
        arr.append(model.model.decode(z))
    res = torch.cat(arr, dim=0)
    vutils.save_image(res, f'../sliding/sample-{dim}.png', normalize=True, nrow=20)
