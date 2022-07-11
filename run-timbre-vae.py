import os
from pathlib import Path

from experiment_timbre import TimbreVAELightningModule
from models import *
from experiment import VAELightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from datasets import CelebAZipDataModule
from pytorch_lightning.plugins import DDPPlugin
from utils import *
from datasets import TimbreDataModule
from torchsummary import summary

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")

config = get_config(parse_args().filename)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = MusicTimbreVAE(**config['model_params'], )
vae = TimbreVAELightningModule(model,
                               config['exp_params'])

## data stuff

data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
# dl = data.train_dataloader()
# fb = next(iter(dl))

# exit()
# data = CelebAZipDataModule(**config["data_params_test"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# data.setup()

trainer = Trainer(logger=tb_logger,
                  callbacks=[
                      LearningRateMonitor(),
                      ModelCheckpoint(save_top_k=100,
                                      dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      # every_n_epochs=1,
                                      every_n_train_steps=10,
                                      monitor="val_loss",
                                      save_last=True),
                  ],
                  strategy=DDPPlugin(find_unused_parameters=True),
                  **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
trainer.fit(vae, datamodule=data)
