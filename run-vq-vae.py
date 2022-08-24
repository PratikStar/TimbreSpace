from __future__ import print_function

import os

import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.signal import savgol_filter
from six.moves import xrange
from torch.utils.data import DataLoader

from models import *

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
epochs = 15000
num_workers=32
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                 ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                   ]))
data_variance = np.var(training_data.data / 255.0)
training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)
validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               num_workers=num_workers,
                               pin_memory=True)

model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

dir_ver = max([int(d.split('_')[1]) for d in next(os.walk('./logs/VQVAECustom/checkpoints'))[1]]) + 1

if not os.path.exists(f'./logs/VQVAECustom/checkpoints/version_{dir_ver}'):
    os.makedirs(f'./logs/VQVAECustom/checkpoints/version_{dir_ver}')

model.train()
train_res_recon_error = []
train_res_perplexity = []

for i in xrange(epochs):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())



    if (i + 1) % 10 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'recon_error': recon_error,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'loss': loss,
        }, f'logs/VQVAECustom/checkpoints/version_{dir_ver}/vq-vae-{i:06}.pt')


# train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
# train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
#
# print(train_res_recon_error_smooth)
# print(train_res_perplexity_smooth)