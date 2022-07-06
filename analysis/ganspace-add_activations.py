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

## Get feature tensors for ganspace. ref: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/3?u=ptrblck
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        print(activation[name].shape)
    return hook
model.model.decoder[0].register_forward_hook(get_activation('0'))
model.model.decoder[1].register_forward_hook(get_activation('1'))
model.model.decoder[2].register_forward_hook(get_activation('2'))
model.model.decoder[3].register_forward_hook(get_activation('3'))


data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
inputs, classes = next(iter(dl))

times = 1
for step, (inputs, classes) in enumerate(dl):
    if step == times:
        break
    f = model.forward(inputs)
