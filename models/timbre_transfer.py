from abc import ABC

import numpy as np
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import collections
from condconv import CondConv2D


class TimbreTransfer(BaseVAE, ABC):

    def __init__(self,
                 config) -> None:
        super(TimbreTransfer, self).__init__()
        self.timbre_encoder_config = config['timbre_encoder']
        self.decoder_config = config['decoder']
        self.config = config

        # Build Timbre Encoder
        modules = []
        in_channels = self.timbre_encoder_config.spectrogram_dims[0]
        h, w = self.timbre_encoder_config.spectrogram_dims[1:]
        print(f"Timbre Encoder Input dims: {self.timbre_encoder_config.spectrogram_dims}")

        for i in range(len(self.timbre_encoder_config.conv2d_channels)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"timbre_encoder_conv2d_{i}", nn.Conv2d(in_channels, out_channels=self.timbre_encoder_config.conv2d_channels[i],
                                                      kernel_size=self.timbre_encoder_config.kernel_size, stride=self.timbre_encoder_config.stride,
                                                      padding=self.timbre_encoder_config.padding)),
                            (f"timbre_encoder_batchNorm2d_{i}", nn.BatchNorm2d(self.timbre_encoder_config.conv2d_channels[i])),
                            (f"timbre_encoder_leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            in_channels = self.timbre_encoder_config.conv2d_channels[i]
            h = math.floor((h + 2 * self.timbre_encoder_config.padding[0] - 1 * (self.timbre_encoder_config.kernel_size[0] - 1) - 1) / self.timbre_encoder_config.stride[0] + 1)
            w = math.floor((w + 2 * self.timbre_encoder_config.padding[1] - 1 * (self.timbre_encoder_config.kernel_size[1] - 1) - 1) / self.timbre_encoder_config.stride[1] + 1)
            print(f"Conv layer dims: ({self.timbre_encoder_config.conv2d_channels[i]}, {h}, {w})")

        conv_layers_output_dim = self.timbre_encoder_config.conv2d_channels[-1] * h * w
        print(f"Encoder conv2d layers output dims: {conv_layers_output_dim}")

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(conv_layers_output_dim, self.timbre_encoder_config.latent_dim)
        self.fc_var = nn.Linear(conv_layers_output_dim, self.timbre_encoder_config.latent_dim)



        # Build Decoder
        in_channels = self.decoder_config.di_spectrogram_dims[0]
        h, w = self.decoder_config.di_spectrogram_dims[1:]
        print(f"Decoder Input dims: {self.decoder_config.di_spectrogram_dims}")

        if self.config.merge_encoding == "sandwich": # z di_b z
            assert h == self.timbre_encoder_config.latent_dim
            w += 2
            print(
                f"Decoder dims after merging timbre encoding ({self.config.merge_encoding}): ({self.decoder_config.conv2d_channels[0]}, {h}, {w})")

            # First decoder transformation to adjust size
            self.merge_encoding_layer = nn.Sequential(
                collections.OrderedDict(
                    [
                        (f"decoder_first_conv2d_merge_layer",
                         nn.Conv2d(in_channels, out_channels=self.decoder_config.conv2d_channels[0],
                                   kernel_size=self.decoder_config.kernel_size,
                                   stride=(1, 1),
                                   padding=(self.decoder_config.padding[0], 1))),
                        (f"decoder_first_batchNorm2d_merge_layer",
                         nn.BatchNorm2d(self.decoder_config.conv2d_channels[0])),
                        (f"decoder_first_leakyReLU_merge_layer",
                         nn.LeakyReLU())
                    ]
                )
            )

            h = math.floor(
                (h + 2 * self.decoder_config.padding[0] - 1 * (self.decoder_config.kernel_size[0] - 1) - 1) / 1 + 1)
            w = math.floor((w + 2 * 1 - 1 * (self.decoder_config.kernel_size[1] - 1) - 1) / 1 + 1)
            assert self.decoder_config.di_spectrogram_dims[1] == h
            assert self.decoder_config.di_spectrogram_dims[2] == w
            print(f"Adjusted decoder input layer dims (upchanneled): ({self.decoder_config.conv2d_channels[0]}, {h}, {w})")

            in_channels = self.decoder_config.conv2d_channels[0]

        elif self.config.merge_encoding == "condconv":
            self.condconv2d = CondConv2D(in_channels=in_channels, out_channels=self.decoder_config.conv2d_channels[0],
                                         kernel_size=self.decoder_config.kernel_size,
                                         stride=self.decoder_config.stride,
                                         num_experts=self.timbre_encoder_config.latent_dim,
                                         padding=self.decoder_config.padding)
            h = math.floor((h + 2 * self.decoder_config.padding[0] - 1 * (self.decoder_config.kernel_size[0] - 1) - 1) / self.decoder_config.stride[0] + 1)
            w = math.floor((w + 2 * self.decoder_config.padding[1] - 1 * (self.decoder_config.kernel_size[1] - 1) - 1) / self.decoder_config.stride[1] + 1)
            print(
                f"Decoder dims after merging timbre encoding ({self.config.merge_encoding}): ({self.decoder_config.conv2d_channels[0]}, {h}, {w})")
            in_channels = self.decoder_config.conv2d_channels[0]

        else:
            raise Exception("merge_encoding not defined")

        modules = []
        for i in range(1, len(self.decoder_config.conv2d_channels)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"decoder_conv2d_{i}", nn.Conv2d(in_channels=in_channels, out_channels=self.decoder_config.conv2d_channels[i],
                                                      kernel_size=self.decoder_config.kernel_size, stride=self.decoder_config.stride,
                                                      padding=self.decoder_config.padding)),
                            (f"decoder_batchNorm2d_{i}", nn.BatchNorm2d(self.decoder_config.conv2d_channels[i])),
                            (f"decoder_leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            in_channels = self.decoder_config.conv2d_channels[i]
            h = math.floor((h + 2 * self.decoder_config.padding[0] - 1 * (self.decoder_config.kernel_size[0] - 1) - 1) / self.decoder_config.stride[0] + 1)
            w = math.floor((w + 2 * self.decoder_config.padding[1] - 1 * (self.decoder_config.kernel_size[1] - 1) - 1) / self.decoder_config.stride[1] + 1)
            print(f"Decoder Conv layer dims: ({self.decoder_config.conv2d_channels[i]}, {h}, {w})")

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(collections.OrderedDict([
            (f"final_sigmoid", nn.Sigmoid())]))


        assert h == self.decoder_config.di_spectrogram_dims[1] and w == self.decoder_config.di_spectrogram_dims[
            2], f"The input {self.decoder_config.di_spectrogram_dims[1:]} and output {[h, w]} dims of the VAE don't match"

    def forward(self, re_a, di_a, re_b, di_b) -> List[Tensor]:
        # Encode the reamped. reparameterize, and get z
        mu, log_var = self.encode(re_a)
        z = self.reparameterize(mu, log_var)

        # transform di input
        adjusted_decoder_input = self.merge_encoding(di_b, z)
        # run decoder
        recons = self.decode(adjusted_decoder_input)
        return [recons, mu, log_var, z]

    def encode(self, re_a: Tensor) -> List[Tensor]:

        result = self.encoder(re_a)
        result = torch.flatten(result, start_dim=1)

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

    def decode(self, input: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(input)
        result = self.final_layer(result)
        return result

    def merge_encoding(self, di_b, z):
        if self.config.merge_encoding == "sandwich":
            z = torch.unsqueeze(z, 1)
            z = torch.unsqueeze(z, 3)
            merged_input = torch.cat((z, di_b, z), 3)
            res = self.merge_encoding_layer(merged_input)
            return res
        if self.config.merge_encoding == "condconv":
            res = self.condconv2d(di_b, z)
            return res
        else:
            raise Exception("merge_encoding not defined")

    def loss_function(self,
                      recons: Tensor,
                      re_b: Tensor,
                      kld_weight: float,
                      log_var: Tensor,
                      mu) -> dict:

        if self.config.loss.function == "L1":
            recons_loss = F.l1_loss(recons, re_b)
        elif self.config.loss.function == "L2":
            recons_loss = F.mse_loss(recons, re_b)
        else:
            raise Exception("Loss function not defined")
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
