import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append('..')
from datasets.celeba import CelebAZipDatasetWithFilter
from experiment import VAELightningModule
from models import *
from utils.utils import *

# Load the model
config = get_config(os.path.join(os.getcwd(), '../configs/vae.yaml'))
chk_path = os.path.join(os.getcwd(), f"../logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model=vae_models[config['model_params']['name']](
                                                    **config['model_params']),
                                                params=config['exp_params'])
# summary(model, (3, 64, 64))

# Get feature tensors for ganspace.
# ref: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/3?u=ptrblck
activation = []


def get_activation():
    def hook(model, input, output):
        # print(input[0].shape) # Because the input is transformed to a tuple
        # print(output.shape)
        activation.append(output.detach())

    return hook


# applying to the first linear layer
handle = model.model.decoder_input.register_forward_hook(get_activation())

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.CenterCrop(148),
                                       transforms.Resize(64),
                                       transforms.ToTensor(), ])
ds = CelebAZipDatasetWithFilter('../../../data/celeba', ('Male', -1),
                                transform=train_transforms)
dl = DataLoader(
    ds,
    batch_size=64,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)
iterdl = iter(dl)
x, y, k = next(iterdl)
# for testing
f = model.forward(x)

n = 1000
layer = 'decoder_input'
