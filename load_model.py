import argparse
import os
import yaml
from dataset import CelebAZipDataModule
from experiment import VAEXperiment
from models import *

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_0/checkpoints/last.ckpt")
model = VAEXperiment.load_from_checkpoint(checkpoint_path=chk_path,
                                        map_location=torch.device('cpu'),
                                          vae_model=VanillaVAE(**config['model_params']),
                                        params=config['exp_params'])

data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
inputs, classes = next(iter(dl))

# datapoint = data.train_dataset.__getitem__(45)

print(inputs.size())
print(classes.size())
f = model.forward(inputs)
print(f[0].size())
print(f[1].size())