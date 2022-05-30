import os
import yaml
from dataset import CelebAZipDataModule
from experiment import VAELightningModule
from models import *
from utils import *
import csv
# For interactive use
# exec(open("./load_model.py").read())

# config = get_config(parse_args())

config = get_config(os.path.join(os.getcwd(), 'configs/vae.yaml'))

chk_path = os.path.join(os.getcwd(), f"logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
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


# For generating embeddings
times = 50
for step, (inputs, classes) in enumerate(dl):
    if step == times:
        break
    f = model.forward(inputs)
    batch_size = config['data_params']['train_batch_size']

    # with open("embeddings.csv", 'a', newline='') as f_output:
    #     tsv_output = csv.writer(f_output, delimiter=',')
    #     tsv_output.writerows(f[4].tolist())
    #
    # with open("attrs.csv", 'a', newline='') as f_output:
    #     tsv_output = csv.writer(f_output, delimiter=',')
    #     tsv_output.writerows(classes.tolist())

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


