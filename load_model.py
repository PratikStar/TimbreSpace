import os
import yaml
from torch.utils.data import DataLoader

from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from playing.dataset import CelebAZipDatasetWithFilter
from utils import *
import torchvision.utils as vutils
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


## For saving images
# z = torch.randn(128)
# s = model.model.decode(z)
#
# vutils.save_image(s, f'sliding/sample-OG.png', normalize=True)
# for dim in range(0, 128):
#     arr = []
#     for i in np.arange(-10, 10, 1):
#         z[dim] = i
#         arr.append(model.model.decode(z))
#     res = torch.cat(arr, dim=0)
#     vutils.save_image(res, f'sliding/sample-{dim}.png', normalize=True, nrow=20)

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
times = 2
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
    vutils.save_image(dec, f'samples/sample-male-{step}.png', normalize=True)
# iterdl = iter(dl)
# x, y, k = next(iterdl)