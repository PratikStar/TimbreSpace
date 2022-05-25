import os
import yaml
from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from utils import *

# config = get_config(parse_args())

config = get_config(os.path.join(os.getcwd(), 'configs/vq_vae.yaml'))

chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_4/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model= vae_models[config['model_params']['name']](**config['model_params']),
                                                params=config['exp_params'])
# Same thing using torch load
# checkpoint = torch.load(chk_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# print(checkpoint.keys())

data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
inputs, classes = next(iter(dl))
f = model.forward(inputs)

#
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


