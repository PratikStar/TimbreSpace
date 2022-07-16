import os
from abc import ABC
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim

from models import BaseVAE, MusicTimbreVAE, MusicVAE
from models.types_ import *
from pytorch_lightning.utilities.types import _METRIC_COLLECTION, EPOCH_OUTPUT, STEP_OUTPUT


class MusicVAELightningModule(pl.LightningModule, ABC):

    def __init__(self,
                 vae_model: MusicVAE, # contains Music vae
                 config: dict,
                 # config_dump: dict, # This is for logging
                 ) -> None:
        super(MusicVAELightningModule, self).__init__()
        self.save_hyperparameters()

        # print(config)
        self.model = vae_model
        self.config = config['exp_params']
        self.config_dump = config
        # self.config = config
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.config['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch_item, batch_idx, optimizer_idx=0):
        # print(f'\n=== Training step. batchidx: {batch_idx}, optimizeridx: {optimizer_idx} ===')
        batch, batch_di, key, offset = batch_item
        batch = torch.squeeze(batch, 0)
        batch_di = torch.squeeze(batch_di, 0)

        print(f"Training step: {batch.type()}")
        print(f"Training step current device: {self.curr_device}")
        # print(f"batch: {batch.shape}, batch_di: {batch_di.shape}, key: {key}, offset: {offset}")
        self.curr_device = batch.device
        print(f"Training step current device: {self.curr_device}")

        music_results = self.model.forward(batch)
        music_train_loss = self.model.loss_function(*music_results, batch_di,
                                              M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in music_train_loss.items()}, sync_dist=True)
        # print(music_train_loss)
        return music_train_loss['loss']


    def validation_step(self, batch_item, batch_idx, ):
        # print(f'\n=== Validation step. batchidx: {batch_idx} ===')
        batch, batch_di, key, offset = batch_item
        print(f"Validation step: {batch.type()}")
        print(f"Validation step current device: {self.curr_device}")

        batch = torch.squeeze(batch, 0)
        batch_di = torch.squeeze(batch_di, 0)

        self.curr_device = batch.device
        # print(type(batch.device))

        music_results = self.model.forward(batch)
        music_val_loss = self.model.loss_function(*music_results, batch_di,
                                              M_N=self.config['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=None,
                                              batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in music_val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):

        music_optimizer = optim.Adam(self.model.parameters(),
                               lr=self.config['LR'],
                               weight_decay=self.config['weight_decay'])

        return music_optimizer

    def on_validation_start(self) -> None:
        # Get sample reconstruction image
        print("on_validation_end")
        batch, batch_di, key, offset = next(iter(self.trainer.datamodule.val_dataloader()))
        batch = torch.squeeze(batch, 0).to(self.curr_device)

        print(f"Min in input: {batch.min()}")
        di_recons, _, _, _, z = self.model.forward(batch)

        di_recons = di_recons.detach().cpu().numpy()
        batch_di = torch.squeeze(batch_di, 0).cpu().numpy()

        # for i in range(di_recons.shape[0]):
        #     self.trainer.datamodule.dataset.preprocessing_pipeline.visualizer.visualize_multiple(
        #         [batch_di[i, 0, :, :], di_recons[i, 0, :, :]],
        #         ['original', 'recons'],
        #         "DI.wav", f"offset - {float(offset):0.2f} - {i}",
        #         file_dir=Path(self.trainer.logger.log_dir) / 'recons')

        self.trainer.datamodule.dataset.preprocessing_pipeline.visualizer.visualize_multiple1(
            batch_di[:,0,:,:], di_recons[:,0,:,:],
            file_dir=Path(self.trainer.logger.log_dir) / 'recons')