import csv
import os

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.celeba import CelebAZipDatasetWithFilter
from experiment import VAELightningModule
from models import *
from utils.utils import *

# Load model
config = get_config(os.path.join(os.getcwd(), '../configs/vae.yaml'))
chk_path = os.path.join(os.getcwd(), f"../logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model=vae_models[config['model_params']['name']](
                                                    **config['model_params']),
                                                params=config['exp_params'])

# Load Dataset
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(148),
                                       transforms.Resize(64),
                                       transforms.ToTensor(), ])
## Repeat for Female. Female = (Male, -1)
ds = CelebAZipDatasetWithFilter('../../../data/celeba', ('Male', -1),
                                transform=train_transforms)
dl = DataLoader(
    ds,
    batch_size=64,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)

times = 2
samples_dir = '../samples'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

# Encode the data and save og vs recons
for step, (x, y, k) in enumerate(dl):
    if step == times:
        break
    f = model.forward(x)
    batch_size = config['data_params']['train_batch_size']

    with open("female.csv", 'a') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')
        e = f[4].cpu().detach().numpy()
        tsv_output.writerows(e)
    dec = model.model.decode(f[4])

    vutils.save_image(x, f'../samples/female-{step}-og.png', normalize=True)
    vutils.save_image(dec, f'../samples/female-{step}-recons.png', normalize=True)
