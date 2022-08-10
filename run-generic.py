import os
from pathlib import Path

from experiment_music import MusicVAELightningModule
from experiment_timbre_transfer_flatten import TimbreTransferFlattenLM
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
import wandb

print(f"torch: {torch.__version__}")
print(f"CUDA #devices: {torch.cuda.device_count()}")

device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
print(f"Device: {device}")

config = get_config(parse_args().filename)
print(config)
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

## data stuff
data = TimbreDataModule(config.data_params, pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
dl = data.train_dataloader()
fb = next(iter(dl))

# model stuff
wandb.init(**config['wandb'])
lightning_module = None
if 'load_path' in config['model_params']:
    print(f"Loading model from {config['model_params']['load_path']}")
    chk_path = os.path.join(os.getcwd(), config['model_params']['load_path'])
    lightning_module = TimbreTransferFlattenLM.load_from_checkpoint(checkpoint_path=chk_path,
                                                             map_location=torch.device(device))
else:
    model = vae_models[config['model_params']['name']](config['model_params'])
    lightning_module = TimbreTransferFlattenLM(model,
                                        config=config)

trainer = Trainer(logger=tb_logger,
                  # check_val_every_n_epoch=5,
                  callbacks=[
                      LearningRateMonitor(),
                      ModelCheckpoint(dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      **config['exp_params']['model_checkpoint']),
                  ],
                  strategy=DDPPlugin(find_unused_parameters=False),
                  **config['trainer_params'])
# exit()
print(f"======= Training {config['model_params']['name']} =======")
trainer.fit(lightning_module, datamodule=data)
