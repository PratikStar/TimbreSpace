import os
import random
from abc import ABC
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from pytorch_lightning.loggers import WandbLogger
from torch import optim

from datasets import TimbreDataModule
from models import TimbreTransfer
from models.types_ import *
from pytorch_lightning.utilities.types import _METRIC_COLLECTION, EPOCH_OUTPUT, STEP_OUTPUT
from utils import re_nest_configs, merge
from prodict import Prodict
from pytorch_lightning.loggers import TensorBoardLogger


class TimbreTransferLM(pl.LightningModule, ABC):

    def __init__(self,
                 model: TimbreTransfer,
                 config: Prodict,
                 ) -> None:
        super(TimbreTransferLM, self).__init__()
        self.save_hyperparameters()

        print(model)

        self.model = model
        self.config = config
        # print(self.config)

        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.config['exp_params']['retain_first_backpass']
        except:
            pass

    def get_data(self):
        return self.data
    def training_step(self, batch_item, batch_idx, optimizer_idx=0):
        # print(f'\n=== Training step. batchidx: {batch_idx}, optimizeridx: {optimizer_idx} ===')
        re_a, di_a, re_b, di_b = self.create_input_batch(batch_item)

        self.curr_device = re_b.device
        # print(type(batch.device))

        recons, mu, log_var, _ = self.model.forward(re_a, di_a, re_b, di_b)
        music_train_loss = self.model.loss_function(recons=recons,
                                                    re_b=re_b,
                                                    kld_weight=self.config['exp_params']['kld_weight'],
                                                    mu=mu,
                                                    log_var=log_var
                                                    )
        self.log('epoch', self.trainer.current_epoch)
        for key, val in music_train_loss.items():
            self.log(key, val)
        return music_train_loss['loss']

    def validation_step(self, batch_item, batch_idx, ):
        # print(f'\n=== Validation step. batchidx: {batch_idx} ===')
        re_a, di_a, re_b, di_b = self.create_input_batch(batch_item)

        self.curr_device = re_b.device

        recons, mu, log_var, _ = self.model.forward(re_a, di_a, re_b, di_b)
        music_val_loss = self.model.loss_function(recons=recons,
                                                  re_b=re_b,
                                                  kld_weight=self.config['exp_params']['kld_weight'],
                                                  mu=mu,
                                                  log_var=log_var
                                                  )
        for key, val in music_val_loss.items():
            self.log(f"val_{key}", val)

    def configure_optimizers(self):

        music_optimizer = optim.Adam(self.model.parameters(),
                                     lr=self.config['exp_params']['LR'],
                                     weight_decay=self.config['exp_params']['weight_decay'])

        return music_optimizer

    def on_validation_end(self) -> None:
        # Get sample reconstruction image
        batch_item = next(iter(self.trainer.datamodule.val_dataloader())) # this does not return current device correctly
        key = batch_item[-2].detach().to("cpu").numpy()[0]
        re_a, di_a, re_b, di_b = self.create_input_batch(batch_item)
        recons, mu, log_var, _ = self.model.forward(re_a.to(self.curr_device), di_a.to(self.curr_device), re_b.to(self.curr_device), di_b.to(self.curr_device))

        di_b = di_b.detach().to("cpu").numpy()
        re_b = re_b.detach().to("cpu").numpy()
        recons = torch.squeeze(recons, 0).to("cpu").numpy()

        spectrograms = [di_b[:, 0, :, :], re_b[:, 0, :, :], recons[:, 0, :, :]]

        if type(self.trainer.logger) == WandbLogger:
            recons_dir = Path(self.trainer.logger.save_dir) / 'recons'
        elif type(self.trainer.logger) == TensorBoardLogger:
            recons_dir = Path(self.trainer.logger.log_dir) / 'recons'

        self.trainer.datamodule.dataset.preprocessing_pipeline.visualizer.visualize_multiple(
            spectrograms,
            file_dir=recons_dir,
            col_titles=["DI", "Expected reamped", "Reconstructed reamped"],
            filename=f"reconstruction-e_{self.trainer.current_epoch}.png",
            title=f"reconstruction for epoch e_{self.trainer.current_epoch}: DI v/s Expected v/s Reconstructed "
                  f"reamped clip (key={key})"
        )
        if type(self.trainer.logger) == WandbLogger:
            try:
                self.trainer.logger.log_image(key=f"epoch-{self.trainer.current_epoch}", images=[
                str(recons_dir / f"reconstruction-e_{self.trainer.current_epoch}.png")])
            except Exception as e:
                print(f"Ignoring Exception while wandb.log_image: {e}")

    def create_input_batch(self, batch):
        batch, batch_di, _, _, _, _ = batch

        batch = torch.squeeze(batch, 0)
        batch_di = torch.squeeze(batch_di, 0)
        b_size = self.config.data_params.batch_size // 2
        re_a = batch[0:b_size, :]
        di_a = batch_di[0:b_size, :]

        re_b = batch[b_size:, :]
        di_b = batch_di[b_size:, :]
        return re_a, di_a, re_b, di_b

    def on_fit_end(self) -> None:
        print("On fit end")
