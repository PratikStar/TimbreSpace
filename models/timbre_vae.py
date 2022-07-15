from abc import ABC

import numpy as np
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import collections



class TimbreVAE(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 music_latent_dim: int,
                 hidden_dims: List[int],
                 timbre_latent_converge: str,
                 in_channels: int = 1,
                 **kwargs) -> None:
        super(TimbreVAE, self).__init__()

        self.latent_converge = timbre_latent_converge
        # Build Timbre Encoder
        modules = []
        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"conv2d_{i}", nn.Conv2d(in_channels, out_channels=hidden_dims[i],
                                                      kernel_size=3, stride=2, padding=1)),
                            # with this, height and width halves in each iteration
                            (f"batchNorm2d_{i}", nn.BatchNorm2d(hidden_dims[i])),
                            (f"leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            # Non-batched: (1, 256, 64) -> (32, 128, 32) -> (64, 64, 16) -> (128, 32, 8) -> (256, 16, 4) -> (512, 8, 2)
            in_channels = hidden_dims[i]

        # Note: manually update the input spectrogram shape here, now, 256 x 64
        # the input H and W has been halved len(hidden_dims) times and the number of channels is hidden_dims[-1]
        conv_layers_output_dim = int(
            (256 / math.pow(2, len(hidden_dims))) * (64 / math.pow(2, len(hidden_dims))) * hidden_dims[-1])

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(conv_layers_output_dim, latent_dim)
        self.fc_var = nn.Linear(conv_layers_output_dim, latent_dim)

        # Build Timbre Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim + music_latent_dim, conv_layers_output_dim)
        # Shape is (C, H, W): product of (512, 8, 2)

        for i in range(len(hidden_dims) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"convTranspose2d_{len(hidden_dims) - i}", nn.ConvTranspose2d(hidden_dims[i],
                                                                                           hidden_dims[i - 1],
                                                                                           kernel_size=3,
                                                                                           stride=2,
                                                                                           padding=1,
                                                                                           output_padding=1)),
                            # H & W essentially doubles
                            (f"batchnorm2d_{len(hidden_dims) - i}", nn.BatchNorm2d(hidden_dims[i - 1])),
                            (f"leakyReLU_{len(hidden_dims) - i}", nn.LeakyReLU()),
                        ]
                    )
                )
            )
            # (512, 8, 2) -> (256, 16, 4) -> (128, 32, 8) -> (64, 64, 16) -> (32, 128, 32)

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(collections.OrderedDict([
            (f"final_convTranspose2d", nn.ConvTranspose2d(hidden_dims[0],
                                                          hidden_dims[0],
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1,
                                                          output_padding=1)),  # output shape is: (32, 256, 64)
            (f"final_batchNorm2d", nn.BatchNorm2d(hidden_dims[0])),
            (f"final_leakyReLU", nn.LeakyReLU()),
            (f"final_Conv2d", nn.Conv2d(hidden_dims[0], out_channels=1,
                                        kernel_size=3, padding=1)),  # output shape is: (1, 256, 64)
            (f"final_tanH", nn.Tanh())]))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, z_music: np.ndarray) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]

        Args:
            z_music:
            z_music:
        """
        # z_music.requires_grad = False
        # print(f"In timbrevae.decode, shape of z: {z.shape} ")
        # print(z)
        # print(z.shape)
        # print(z_music)
        # print(z_music.shape)
        if self.latent_converge == "first":
            z = z[0].repeat(8, 1)
        elif self.latent_converge == "mean":
            z = z.mean(dim=0)
        elif self.latent_converge == "max":
            z = z.max(dim=0).values

        result = self.decoder_input(torch.cat((z, torch.tensor(z_music, device=self.device)), dim=1))
        result = result.view(-1, 512, 8, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def set_device(self, device):
        self.device = device


class MusicTimbreVAE(BaseVAE):

    def __init__(self,
                 music_latent_dim: int,
                 timbre_latent_dim: int,
                 # device: str,
                 **kwargs) -> None:
        super(MusicTimbreVAE, self).__init__()

        self.music_latent_dim = music_latent_dim
        self.timbre_latent_dim = timbre_latent_dim

        hidden_dims = [32, 64, 128, 256, 512]
        self.music_vae = MusicVAE(latent_dim=music_latent_dim, hidden_dims=hidden_dims)
        self.timbre_vae = TimbreVAE(latent_dim=timbre_latent_dim, music_latent_dim=music_latent_dim,
                                    hidden_dims=hidden_dims, **kwargs)

    def forward_music(self, input: Tensor, **kwargs) -> List[Tensor]:
        music_mu, music_log_var = self.music_vae.encode(input)
        music_z = self.music_vae.reparameterize(music_mu, music_log_var)
        return [self.music_vae.decode(music_z), input, music_mu, music_log_var, music_z]

    def forward_timbre(self, input: Tensor, music_z: np.ndarray, **kwargs) -> List[Tensor]:
        timbre_mu, timbre_log_var = self.timbre_vae.encode(input)
        timbre_z = self.timbre_vae.reparameterize(timbre_mu, timbre_log_var)
        return [self.timbre_vae.decode(timbre_z, music_z), input, timbre_mu, timbre_log_var, timbre_z]

    def loss_function_music(self,
                            *args,
                            **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def loss_function_timbre(self,
                             *args,
                             **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def set_device(self, device):
        self.timbre_vae.set_device(device)
