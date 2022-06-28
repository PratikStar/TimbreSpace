import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from experiment import VAELightningModule
from models import vae_models
from utils.utils import get_config
import torch
import torchvision.utils as vutils

with open('male.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    a = []
    wait = 1000
    for row in csvreader:
        wait -= 1
        em = [float(i) for i in row[0].split(',')]
        a.append(em)
        # print(', '.join(row))
        if wait == 0:
            break
    arr_male = np.array(a)

with open('female.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    a = []
    wait = 1000
    for row in csvreader:
        wait -= 1
        em = [float(i) for i in row[0].split(',')]
        a.append(em)
        # print(', '.join(row))
        if wait == 0:
            break
    arr_female = np.array(a)

mean_male = np.mean(arr_male, axis=0)
std_male = np.std(arr_male, axis=0)

mean_female = np.mean(arr_female, axis=0)
std_female = np.std(arr_female, axis=0)


# Plot
ax = plt.gca()
ax.set_title('centered spines')
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot(mean_male, np.arange(0, 128), 'ro', marker=".", markersize=5, label='Mean of Male')
ax.plot(mean_female, np.arange(0, 128), 'go', marker=".", markersize=5, label='Mean of Female')
ax.set_ylabel('Dimension')
ax.set_xlabel('Mean')

plt.show()

# Load model & Decode
config = get_config(os.path.join(os.getcwd(), '../configs/vae.yaml'))
chk_path = os.path.join(os.getcwd(), f"../logs/{config['model_params']['name']}/version_12/checkpoints/last.ckpt")
model = VAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                map_location=torch.device('cpu'),
                                                vae_model=vae_models[config['model_params']['name']](
                                                    **config['model_params']),
                                                params=config['exp_params'])
dec = model.model.decode(torch.tensor(mean_male, dtype=torch.float))
vutils.save_image(dec, f'mean-male.png', normalize=True)
