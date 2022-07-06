from abc import ABC

import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import collections


class MusicVAE(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 in_channels: int = 1,
                 **kwargs) -> None:
        super(MusicVAE, self).__init__()

        # Build Music Encoder
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
        # the input H and W has be halved len(hidden_dims) times and the number of channels is hidden_dims[-1]
        conv_layers_output_dim = int(
            (256 / math.pow(2, len(hidden_dims))) * (64 / math.pow(2, len(hidden_dims))) * hidden_dims[-1])

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(conv_layers_output_dim, latent_dim)
        self.fc_var = nn.Linear(conv_layers_output_dim, latent_dim)

        # Build Music Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, conv_layers_output_dim)
        # Shape is (C, H, W): (512, 8, 2)

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
            (f"final_convTranspose2d", nn.ConvTranspose2d(hidden_dims[-1],
                                                          hidden_dims[-1],
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1,
                                                          output_padding=1)),  # output shape is: (32, 256, 64)
            (f"final_batchNorm2d", nn.BatchNorm2d(hidden_dims[-1])),
            (f"final_leakyReLU", nn.LeakyReLU()),
            (f"final_Conv2d", nn.Conv2d(hidden_dims[-1], out_channels=1,
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

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
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


class TimbreVAE(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 in_channels: int = 1,
                 **kwargs) -> None:
        super(TimbreVAE, self).__init__()

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
        self.decoder_input = nn.Linear(latent_dim, conv_layers_output_dim)
        # Shape is (C, H, W): (512, 8, 2)

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
            (f"final_convTranspose2d", nn.ConvTranspose2d(hidden_dims[-1],
                                                                 hidden_dims[-1],
                                                                 kernel_size=3,
                                                                 stride=2,
                                                                 padding=1,
                                                                 output_padding=1)),  # output shape is: (32, 256, 64)
            (f"final_batchNorm2d", nn.BatchNorm2d(hidden_dims[-1])),
            (f"final_leakyReLU", nn.LeakyReLU()),
            (f"final_Conv2d", nn.Conv2d(hidden_dims[-1], out_channels=1,
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

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
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


class MusicTimbreVAE(BaseVAE):

    def __init__(self,
                 music_latent_dim: int,
                 timbre_latent_dim: int,
                 **kwargs) -> None:
        super(MusicTimbreVAE, self).__init__()

        self.music_latent_dim = music_latent_dim
        self.timbre_latent_dim = timbre_latent_dim

        hidden_dims = [32, 64, 128, 256, 512]
        self.music_vae = MusicVAE(latent_dim=music_latent_dim, hidden_dims=hidden_dims)
        self.timbre_vae = TimbreVAE(latent_dim=music_latent_dim, hidden_dims=hidden_dims)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.music_vae.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.music_vae.fc_mu(result)
        log_var = self.music_vae.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.music_vae.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.music_vae.decoder(result)
        result = self.music_vae.final_layer(result)
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
        mu, log_var = self.music_vae.encode(input)
        z = self.music_vae.reparameterize(mu, log_var)
        return [self.music_vae.decode(z), input, mu, log_var, z]

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

    # For inference
    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    # For inference
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
