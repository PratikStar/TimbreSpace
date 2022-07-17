from abc import ABC

import numpy as np
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import collections


class MusicVAEFlat(BaseVAE, ABC):

    def __init__(self,
                 latent_dim: int,
                 conv2d_channels: List[int],
                 spectrogram_dims: List,
                 stride=(2, 2),
                 kernel_size=(3, 3),
                 padding=(1, 1),
                 output_padding=(1, 1),
                 loss=None,
                 **kwargs) -> None:
        super(MusicVAEFlat, self).__init__()
        self.spectrogram_dims = spectrogram_dims
        self.latent_dim = latent_dim
        self.conv2d_channels = conv2d_channels
        self.stride = tuple(stride)
        self.kernel_size = tuple(kernel_size)
        self.padding = tuple(padding)
        self.output_padding = tuple(output_padding)
        self.loss_func = loss['function']


        # Build Music Encoder
        modules = []
        in_channels = self.spectrogram_dims[0]
        h, w = self.spectrogram_dims[1:]
        print(f"Input dims: {self.spectrogram_dims}")

        for i in range(len(self.conv2d_channels) - 1):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"conv2d_{i}", nn.Conv2d(in_channels, out_channels=self.conv2d_channels[i],
                                                      kernel_size=self.kernel_size, stride=self.stride,
                                                      padding=self.padding)),
                            (f"batchNorm2d_{i}", nn.BatchNorm2d(self.conv2d_channels[i])),
                            (f"leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            in_channels = self.conv2d_channels[i]
            h = math.floor((h + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
            w = math.floor((w + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
            print(f"Conv layer dims: ({self.conv2d_channels[i]}, {h}, {w})")

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Sequential(
            collections.OrderedDict(
                [
                    (f"conv2d_fc_mu", nn.Conv2d(self.conv2d_channels[-2], out_channels=self.conv2d_channels[-1],
                                                kernel_size=self.kernel_size, stride=self.stride,
                                                padding=self.padding)),
                    (f"batchNorm2d_fc_mu", nn.BatchNorm2d(self.conv2d_channels[-1])),
                    (f"leakyReLU_fc_mu", nn.LeakyReLU())
                ]
            )
        )
        self.fc_var = nn.Sequential(
            collections.OrderedDict(
                [
                    (f"conv2d_fc_var", nn.Conv2d(self.conv2d_channels[-2], out_channels=self.conv2d_channels[-1],
                                                 kernel_size=self.kernel_size, stride=self.stride,
                                                 padding=self.padding)),
                    (f"batchNorm2d_fc_var", nn.BatchNorm2d(self.conv2d_channels[-1])),
                    (f"leakyReLU_fc_var", nn.LeakyReLU())
                ]
            )
        )
        h = math.floor((h + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w = math.floor((w + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        print(f"Conv (mu, logvar) layer dims: ({self.conv2d_channels[-1]}, {h}, {w})")

        # Build Music Decoder
        modules = []
        for i in range(len(self.conv2d_channels) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"convTranspose2d_{len(self.conv2d_channels) - i}",
                             nn.ConvTranspose2d(self.conv2d_channels[i],
                                                self.conv2d_channels[i - 1],
                                                kernel_size=self.kernel_size,
                                                stride=self.stride,
                                                padding=self.padding,
                                                output_padding=self.output_padding)),
                            (f"batchnorm2d_{len(self.conv2d_channels) - i}",
                             nn.BatchNorm2d(self.conv2d_channels[i - 1])),
                            (f"leakyReLU_{len(self.conv2d_channels) - i}", nn.LeakyReLU()),
                        ]
                    )
                )
            )
            h = (h - 1) * self.stride[0] - 2 * self.padding[0] + 1 * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
            w = (w - 1) * self.stride[1] - 2 * self.padding[1] + 1 * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
            print(f"ConvTranspose layer dims: ({self.conv2d_channels[i-1]}, {h}, {w})")

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(collections.OrderedDict([
            (f"final_convTranspose2d", nn.ConvTranspose2d(self.conv2d_channels[0],
                                                          self.conv2d_channels[0],
                                                          kernel_size=self.kernel_size,
                                                          stride=self.stride,
                                                          padding=self.padding,
                                                          output_padding=self.output_padding)),
            (f"final_batchNorm2d", nn.BatchNorm2d(self.conv2d_channels[0])),
            (f"final_leakyReLU", nn.LeakyReLU()),
            (f"final_Conv2d", nn.Conv2d(self.conv2d_channels[0], out_channels=1, stride=1,
                                        kernel_size=3, padding=1)),  # output shape is: (1, 256, 64)
            (f"final_tanH", nn.Sigmoid())]))

        # Final convtranspose2d layer
        h = (h - 1) * self.stride[0] - 2 * self.padding[0] + 1 * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        w = (w - 1) * self.stride[1] - 2 * self.padding[1] + 1 * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        print(f"final ConvTranspose layer dims: ({self.conv2d_channels[0]}, {h}, {w})")
        # Final conv layer
        h = math.floor((h + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / 1 + 1)
        w = math.floor((w + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / 1 + 1)
        print(f"final ConvTranspose layer dims: (1, {h}, {w})")

        assert h == self.spectrogram_dims[1] and w == self.spectrogram_dims[2], f"The input {self.spectrogram_dims[1:]} and output {[h, w]} dims of the VAE don't match"

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

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

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        # result = self.decoder_input(z)
        # result = result.view(-1, self.conv2d_channels[-1],
        #                      int(self.spectrogram_dims[0] / math.pow(2, len(self.conv2d_channels))),
        #                      int(self.spectrogram_dims[1] / math.pow(2, len(self.conv2d_channels))))
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

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
        di = args[5]
        # print(f"recons: {recons.shape}")
        # print(f"di: {di.shape}")

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        if self.loss_func == "L1":
            recons_loss = F.l1_loss(recons, di)
        else:
            recons_loss = F.mse_loss(recons, di)
        # print(f"recons_loss: {recons_loss.shape}, {recons_loss}")
        # print(f"mu: {mu.shape}")
        # print(f"log_var: {log_var.shape}")
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=(1, 2, 3)), dim=0)
        # print(f"kld_loss: {kld_loss.shape}, {kld_loss}")

        loss = recons_loss + kld_weight * kld_loss
        # print(f"Shape of loss: {loss.shape}, {loss}")
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}