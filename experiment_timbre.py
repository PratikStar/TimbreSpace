import os
from abc import ABC

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim

from models import BaseVAE, MusicTimbreVAE
from models.types_ import *


class TimbreVAELightningModule(pl.LightningModule, ABC):

    def __init__(self,
                 vae_model: MusicTimbreVAE, # contains music vae and timbre vae
                 config: dict) -> None:
        super(TimbreVAELightningModule, self).__init__()
        self.save_hyperparameters()  # Added by me, to test

        self.model = vae_model
        self.config = config
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.config['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch_item, batch_idx, optimizer_idx=0):
        print(f'\n=== Training step. batchidx: {batch_idx}, optimizeridx: {optimizer_idx} ===')
        batch, batch_di, key, offset = batch_item
        batch = torch.squeeze(batch, 0)
        # print(f"batch: {batch.shape}, batch_di: {batch_di.shape}, key: {key}, offset: {offset}")
        self.curr_device = batch.device
        self.model.set_device(self.curr_device)

        if optimizer_idx == 0:
            music_results = self.model.forward_music(batch)
            self.z_music = music_results[4].cpu().detach().numpy()
            # print(f"z_music: {self.z_music.shape}")
            music_train_loss = self.model.loss_function_music(*music_results,
                                                  M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx=batch_idx)
            self.log_dict({key: val.item() for key, val in music_train_loss.items()}, sync_dist=True)
            print(music_train_loss)
            return music_train_loss['loss']

        if optimizer_idx == 1:
            timbre_results = self.model.forward_timbre(batch, self.z_music)
            timbre_train_loss = self.model.loss_function_timbre(*timbre_results,
                                                  M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx=batch_idx)

            self.log_dict({key: val.item() for key, val in timbre_train_loss.items()}, sync_dist=True)
            print(timbre_train_loss)
            return timbre_train_loss['loss']

    def validation_step(self, batch_item, batch_idx, optimizer_idx):
        print(f'\n=== Validation step. batchidx: {batch_idx}, optimizeridx: {optimizer_idx} ===')
        batch, batch_di, key, offset = batch_item
        batch = torch.squeeze(batch, 0)
        self.curr_device = batch.device
        self.model.set_device(self.curr_device)

        if optimizer_idx == 0:
            music_results = self.model.forward_music(batch)
            self.z_music = music_results[4].cpu().detach().numpy()
            # print(f"z_music: {self.z_music.shape}")
            music_val_loss = self.model.loss_function_music(*music_results,
                                                  M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx=batch_idx)
            self.log_dict({f"val_{key}": val.item() for key, val in music_val_loss.items()}, sync_dist=True)

        if optimizer_idx == 1:
            timbre_results = self.model.forward_timbre(batch, self.z_music)
            timbre_val_loss = self.model.loss_function_timbre(*timbre_results,
                                                  M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx=batch_idx)

            self.log_dict({f"val_{key}": val.item() for key, val in timbre_val_loss.items()}, sync_dist=True)


    def configure_optimizers(self):

        optims = []

        music_optimizer = optim.Adam(self.model.music_vae.parameters(),
                               lr=self.config['LR'],
                               weight_decay=self.config['weight_decay'])

        timbre_optimizer = optim.Adam(self.model.timbre_vae.parameters(),
                               lr=self.config['LR'],
                               weight_decay=self.config['weight_decay'])
        optims.append(music_optimizer)
        optims.append(timbre_optimizer)
        return optims
