import os
import yaml
from torch.utils.data import DataLoader

from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from playing.dataset import CelebAZipDatasetWithFilter
from utils import *
import csv
from torchvision import transforms


config = get_config(os.path.join(os.getcwd(), 'configs/vae.yaml'))

chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model= vae_models[config['model_params']['name']](**config['model_params']),
                                                params=config['exp_params'])

## For generating embeddings
# data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# data.setup()
# dl = data.train_dataloader()
# inputs, classes = next(iter(dl))

# times = 50
# for step, (inputs, classes) in enumerate(dl):
#     if step == times:
#         break
#     f = model.forward(inputs)
#     batch_size = config['data_params']['train_batch_size']
#
#     with open("for_umap.csv", 'a') as f_output:
#         tsv_output = csv.writer(f_output, delimiter=',')
#         c = classes.cpu().detach().numpy()
#         e = f[4].cpu().detach().numpy()
#         r = np.concatenate((c, e), axis=1)
#         tsv_output.writerows(r)




train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(148),
                                       transforms.Resize(64),
                                       transforms.ToTensor(), ])
ds = CelebAZipDatasetWithFilter('../../data/celeba', ('Male', 1),
                                transform=train_transforms)

dl = DataLoader(
    ds,
    batch_size=64,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)
times = 50
for step, (x, y, k) in enumerate(dl):
    if step == times:
        break
    f = model.forward(x)
    batch_size = config['data_params']['train_batch_size']

    with open("male.csv", 'a') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')
        c = classes.cpu().detach().numpy()
        e = f[4].cpu().detach().numpy()
        r = np.concatenate((c, e), axis=1)
        tsv_output.writerows(r)

iterdl = iter(dl)
x, y, k = next(iterdl)