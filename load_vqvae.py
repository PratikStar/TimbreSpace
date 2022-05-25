import torch
import torch.optim as optim

from models import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = get_config(parse_args())

batch_size = 256
epochs = 150

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('/work/gk77/k77021/repos/PyTorch-VAE/logs/VQVAECustom/checkpoints/vq-vae-140.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(model)
