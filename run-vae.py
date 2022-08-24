import os
import yaml
from pathlib import Path
from models import *
from experiment import VAELightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets import CelebAZipDataModule
from pytorch_lightning.plugins import DDPPlugin


from utils import *

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")

config = get_config(parse_args().filename)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
vae = VAELightningModule(model,
                         config['exp_params'])

data = CelebAZipDataModule(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# data.setup()

trainer = Trainer(logger=tb_logger,
                  callbacks=[
                      LearningRateMonitor(),
                      ModelCheckpoint(save_top_k=100,
                                      dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      every_n_epochs=25,
                                      monitor="val_loss",
                                      save_last=True),
                  ],
                  # strategy=DDPPlugin(find_unused_parameters=False),
                  **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
trainer.fit(vae, datamodule=data)
