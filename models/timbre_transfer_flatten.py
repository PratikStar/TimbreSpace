from abc import ABC

import numpy as np
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
import collections


class TimbreTransferFlatten(nn.Module):

    def __init__(self,
                 config) -> None:
        super(TimbreTransferFlatten, self).__init__()

        self.config = config

        # Build Timbre Encoder
        modules = []
        in_channels = self.config.timbre_encoder.spectrogram_dims[0]
        h, w = self.config.timbre_encoder.spectrogram_dims[1:]
        print(f"Timbre Encoder Input dims: {self.config.timbre_encoder.spectrogram_dims}")

        for i in range(len(self.config.timbre_encoder.conv2d_channels)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"timbre_conv2d_{i}",
                             nn.Conv2d(in_channels, out_channels=self.config.timbre_encoder.conv2d_channels[i],
                                       kernel_size=self.config.timbre_encoder.kernel_size,
                                       stride=self.config.timbre_encoder.stride,
                                       padding=self.config.timbre_encoder.padding)),
                            (f"timbre_batchNorm2d_{i}", nn.BatchNorm2d(self.config.timbre_encoder.conv2d_channels[i])),
                            (f"timbre_leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            in_channels = self.config.timbre_encoder.conv2d_channels[i]
            h = math.floor((h + 2 * self.config.timbre_encoder.padding[0] - 1 * (
                    self.config.timbre_encoder.kernel_size[0] - 1) - 1) / self.config.timbre_encoder.stride[0] + 1)
            w = math.floor((w + 2 * self.config.timbre_encoder.padding[1] - 1 * (
                    self.config.timbre_encoder.kernel_size[1] - 1) - 1) / self.config.timbre_encoder.stride[1] + 1)
            print(f"Timbre Conv layer dims: ({self.config.timbre_encoder.conv2d_channels[i]}, {h}, {w})")

        timbre_conv_layers_output_dim = self.config.timbre_encoder.conv2d_channels[-1] * h * w
        print(f"Timbre Encoder conv2d layers output dims: {timbre_conv_layers_output_dim}")

        self.timbre_encoder = nn.Sequential(*modules)
        self.fc_mu_t = nn.Linear(timbre_conv_layers_output_dim, self.config.timbre_encoder.latent_dim)
        self.fc_var_t = nn.Linear(timbre_conv_layers_output_dim, self.config.timbre_encoder.latent_dim)

        print(f"Timbre Encoder Last Linear layer: {timbre_conv_layers_output_dim} --> {self.config.timbre_encoder.latent_dim}")

        # Build Music Encoder
        modules = []
        in_channels = self.config.music_encoder.spectrogram_dims[0]
        h, w = self.config.music_encoder.spectrogram_dims[1:]
        print(f"Music Encoder Input dims: {self.config.music_encoder.spectrogram_dims}")

        for i in range(len(self.config.music_encoder.conv2d_channels)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"music_conv2d_{i}",
                             nn.Conv2d(in_channels, out_channels=self.config.music_encoder.conv2d_channels[i],
                                       kernel_size=self.config.music_encoder.kernel_size,
                                       stride=self.config.music_encoder.stride,
                                       padding=self.config.music_encoder.padding)),
                            (f"music_batchNorm2d_{i}", nn.BatchNorm2d(self.config.music_encoder.conv2d_channels[i])),
                            (f"music_leakyReLU_{i}", nn.LeakyReLU())
                        ]
                    )
                )
            )
            in_channels = self.config.music_encoder.conv2d_channels[i]
            h = math.floor((h + 2 * self.config.music_encoder.padding[0] - 1 * (
                    self.config.music_encoder.kernel_size[0] - 1) - 1) / self.config.music_encoder.stride[0] + 1)
            w = math.floor((w + 2 * self.config.music_encoder.padding[1] - 1 * (
                    self.config.music_encoder.kernel_size[1] - 1) - 1) / self.config.music_encoder.stride[1] + 1)
            print(f"Music Conv layer dims: ({self.config.music_encoder.conv2d_channels[i]}, {h}, {w})")

        music_conv_layers_output_dim = self.config.music_encoder.conv2d_channels[-1] * h * w
        print(f"Music Encoder conv2d layers output dims: {music_conv_layers_output_dim}")

        self.music_encoder = nn.Sequential(*modules)
        self.music_last_conv_layer_h = h
        self.music_last_conv_layer_w = w

        self.fc_mu_m = nn.Linear(music_conv_layers_output_dim, self.config.music_encoder.latent_dim)
        self.fc_var_m = nn.Linear(music_conv_layers_output_dim, self.config.music_encoder.latent_dim)
        print(f"Music Encoder Last Linear layer: {music_conv_layers_output_dim} --> {self.config.music_encoder.latent_dim}")

        print(f"----- Decoder starts -----")

        self.decoder_input = nn.Linear(self.config.music_encoder.latent_dim + self.config.timbre_encoder.latent_dim,
                                       music_conv_layers_output_dim)

        print(f"Decoder Input Linear layer: {self.config.music_encoder.latent_dim + self.config.timbre_encoder.latent_dim} --> {music_conv_layers_output_dim}")

        # Build Decoder
        modules = []
        in_channels = self.config.decoder.conv2d_channels[0]
        h, w = self.music_last_conv_layer_h, self.music_last_conv_layer_w
        print(f"Decoder Input dims: ({in_channels}, {h}, {w})")

        for i in range(1, len(self.config.decoder.conv2d_channels)):
            modules.append(
                nn.Sequential(
                    collections.OrderedDict(
                        [
                            (f"convTranspose2d_{i}", nn.ConvTranspose2d(in_channels,
                                                                        self.config.decoder.conv2d_channels[i],
                                                                        kernel_size=self.config.decoder.kernel_size,
                                                                        stride=self.config.decoder.stride,
                                                                        padding=self.config.decoder.padding,
                                                                        output_padding=self.config.decoder.output_padding)),
                            (f"batchNorm2d_{i}", nn.BatchNorm2d(self.config.decoder.conv2d_channels[i])),
                            (f"leakyReLU_{i}", nn.LeakyReLU()),
                        ]
                    )
                )
            )
            in_channels = self.config.decoder.conv2d_channels[i]

            h = (h - 1) * self.config.decoder.stride[0] - 2 * self.config.decoder.padding[0] + 1 * (self.config.decoder.kernel_size[0] - 1) + self.config.decoder.output_padding[0] + 1
            w = (w - 1) * self.config.decoder.stride[1] - 2 * self.config.decoder.padding[1] + 1 * (self.config.decoder.kernel_size[1] - 1) + self.config.decoder.output_padding[1] + 1
            print(f"Decoder ConvTranspose layer dims: ({self.config.decoder.conv2d_channels[i]}, {h}, {w})")

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(collections.OrderedDict([
            (f"final_convTranspose2d", nn.ConvTranspose2d(self.config.decoder.conv2d_channels[-1],
                                                          self.config.decoder.conv2d_channels[-1],
                                                          kernel_size=self.config.decoder.kernel_size,
                                                          stride=self.config.decoder.stride,
                                                          padding=self.config.decoder.padding,
                                                          output_padding=self.config.decoder.output_padding)),
            (f"final_batchNorm2d", nn.BatchNorm2d(self.config.decoder.conv2d_channels[-1])),
            (f"final_leakyReLU", nn.LeakyReLU()),
            (f"final_Conv2d", nn.Conv2d(self.config.decoder.conv2d_channels[-1], out_channels=1,
                                        kernel_size=self.config.decoder.kernel_size, padding=self.config.decoder.padding)),  # output shape is: (1, 256, 64)
            (f"final_Sigmoid", nn.Sigmoid())]))

        # Final convtranspose2d layer
        h = (h - 1) * self.config.decoder.stride[0] - 2 * self.config.decoder.padding[0] + 1 * (
                    self.config.decoder.kernel_size[0] - 1) + self.config.decoder.output_padding[0] + 1
        w = (w - 1) * self.config.decoder.stride[1] - 2 * self.config.decoder.padding[1] + 1 * (
                    self.config.decoder.kernel_size[1] - 1) + self.config.decoder.output_padding[1] + 1
        print(f"Decoder final ConvTranspose layer dims: ({self.config.decoder.conv2d_channels[i]}, {h}, {w})")

        # Final conv layer

        h = math.floor((h + 2 * self.config.decoder.padding[0] - 1 * (
                self.config.decoder.kernel_size[0] - 1) - 1) / 1 + 1)
        w = math.floor((w + 2 * self.config.decoder.padding[1] - 1 * (
                self.config.decoder.kernel_size[1] - 1) - 1) / 1 + 1)

        print(f"Decoder final Conv layer dims: (1, {h}, {w})")

    def encode_timbre(self, re_a: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.timbre_encoder(re_a)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_t(result)
        log_var = self.fc_var_t(result)

        return [mu, log_var]

    def encode_music(self, di_b: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.music_encoder(di_b)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_m(result)
        log_var = self.fc_var_m(result)

        return [mu, log_var]

    def decode(self, merged_encoding: Tensor) -> Tensor:

        result = self.decoder_input(merged_encoding)
        result = result.view(-1, self.config.music_encoder.conv2d_channels[-1], self.music_last_conv_layer_h, self.music_last_conv_layer_w)
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

    def converge_timbre_latent(self, z):

        if self.config.timbre_encoder.converge_latent == "first":
            z = z[0].repeat(z.shape[0], 1)
        elif self.config.timbre_encoder.converge_latent == "mean":
            z = z.mean(dim=0)
        elif self.config.timbre_encoder.converge_latent == "max":
            z = z.max(dim=0).values
        elif self.config.timbre_encoder.converge_latent == "none":
            pass
        else:
            raise Exception("merge_encoding is not valid")
        return z

    def forward(self, re_a, di_a, re_b, di_b) -> List[Tensor]:
        mu_t, log_var_t = self.encode_timbre(re_a)
        zt = self.reparameterize(mu_t, log_var_t)
        zt = self.converge_timbre_latent(zt)

        mu_m, log_var_m = self.encode_music(di_b)
        zm = self.reparameterize(mu_m, log_var_m)

        # transform di input
        merged_encoding = self.merge_encoding(zm, zt)
        # run decoder
        recons = self.decode(merged_encoding)

        return [recons, mu_t, log_var_t, zt, mu_m, log_var_m, zm]

    def merge_encoding(self, zm, zt):
        if self.config.merge_encoding == "cat":
            merged_encoding = torch.cat((zm, zt), 1)
            return merged_encoding
        else:
            raise Exception("merge_encoding not defined")

    def loss_function(self,
                      recons: Tensor,
                      re_b: Tensor,
                      kld_weight_timbre: float,
                      kld_weight_music: float,
                      log_var_t: Tensor,
                      mu_t: Tensor,
                      log_var_m: Tensor,
                      mu_m: Tensor,
                      ) -> dict:

        if self.config.loss.function == "L1":
            recons_loss = F.l1_loss(recons, re_b)
        elif self.config.loss.function == "L2":
            recons_loss = F.mse_loss(recons, re_b)
        else:
            raise Exception("Loss function not defined")
        kld_loss_timbre = torch.mean(-0.5 * torch.sum(1 + log_var_t - mu_t ** 2 - log_var_t.exp(), dim=1), dim=0)
        kld_loss_music = torch.mean(-0.5 * torch.sum(1 + log_var_m - mu_m ** 2 - log_var_m.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight_timbre * kld_loss_timbre + kld_loss_music * kld_weight_music
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(),
                'KLD_music': -kld_loss_music.detach(),
                'KLD_timbre': -kld_loss_timbre.detach()
                }
