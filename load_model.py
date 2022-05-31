import os
import yaml
from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from utils import *
import csv

config = get_config(os.path.join(os.getcwd(), 'configs/vae.yaml'))

chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model= vae_models[config['model_params']['name']](**config['model_params']),
                                                params=config['exp_params'])

data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
inputs, classes = next(iter(dl))

f = model.forward(inputs)


# For generating embeddings
times = 50
for step, (inputs, classes) in enumerate(dl):
    if step == times:
        break
    f = model.forward(inputs)
    batch_size = config['data_params']['train_batch_size']

    with open("for_umap.csv", 'a') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')
        c = classes.cpu().detach().numpy()
        e = f[4].cpu().detach().numpy()
        # c.extend(.tolist())
        r = np.concatenate((c, e), axis=1)
        # print((c.shape))
        # print((e.shape))
        # print((r.shape))
        # print((c))
        tsv_output.writerows(r)



