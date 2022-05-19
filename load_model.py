import argparse
import os
import yaml
from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from utils import parse_config

config = parse_config()

chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_0/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model=VanillaVAE(**config['model_params']),
                                                params=config['exp_params'])

data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
inputs, classes = next(iter(dl))

# datapoint = data.train_dataset.__getitem__(45)

f = model.forward(inputs)
print(f[0].size())
print(f[1].size())

# Same thing using torch load
checkpoint = torch.load(chk_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(checkpoint.keys())
