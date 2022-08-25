import os
from abc import ABC
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim

from models import TimbreTransfer
from models.types_ import *
from pytorch_lightning.utilities.types import _METRIC_COLLECTION, EPOCH_OUTPUT, STEP_OUTPUT
import wandb
from utils import re_nest_configs
from prodict import Prodict


class TimbreTransferLM(pl.LightningModule, ABC):

    def __init__(self,
                 model: TimbreTransfer,  # contains Music vae
                 config: dict,
                 # config_dump: dict, # This is for logging
                 ) -> None:
        super(TimbreTransferLM, self).__init__()
        self.save_hyperparameters()

        wandb.config.update(re_nest_configs({**self.hparams.config, **wandb.config}))
        wandb.watch(model)
        print(model)

        self.model = model
        self.config = Prodict.from_dict(wandb.config)
        print(self.config)

        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.config['exp_params']['retain_first_backpass']
        except:
            pass

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

        log_dict = {key: val.item() for key, val in music_train_loss.items()}
        log_dict['epoch'] = self.trainer.current_epoch
        wandb.log(log_dict)
        self.log_dict(log_dict, sync_dist=True)
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
        # print(music_val_loss)
        log_dict = {f"val_{key}": val.item() for key, val in music_val_loss.items()}
        log_dict['epoch'] = self.trainer.current_epoch
        wandb.log(log_dict)
        self.log_dict(log_dict, sync_dist=True)

    def configure_optimizers(self):

        music_optimizer = optim.Adam(self.model.parameters(),
                                     lr=self.config['exp_params']['LR'],
                                     weight_decay=self.config['exp_params']['weight_decay'])

        return music_optimizer

    def on_validation_end(self) -> None:
        # Get sample reconstruction image
        batch_item = next(iter(self.trainer.datamodule.val_dataloader())) # this does not return current device correctly
        re_a, di_a, re_b, di_b = self.create_input_batch(batch_item)
        recons, mu, log_var, _ = self.model.forward(re_a.to(self.curr_device), di_a.to(self.curr_device), re_b.to(self.curr_device), di_b.to(self.curr_device))

        di_b = di_b.detach().to("cpu").numpy()
        re_b = re_b.detach().to("cpu").numpy()
        recons = torch.squeeze(recons, 0).to("cpu").numpy()

        spectrograms = [di_b[:, 0, :, :], re_b[:, 0, :, :], recons[:, 0, :, :]]

        self.trainer.datamodule.dataset.preprocessing_pipeline.visualizer.visualize_multiple(
            spectrograms,
            file_dir=Path(self.trainer.logger.log_dir) / 'recons',
            col_titles=["DI", "Expected reamped", "Reconstructed reamped"],
            filename=f"reconstruction-e_{self.trainer.current_epoch}.png",
            title=f"reconstruction for epoch e_{self.trainer.current_epoch}: DI v/s Expected v/s Reconstructed reamped clip"
        )

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
