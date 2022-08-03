import os
import io
from abc import ABC
from typing import List, Optional, Sequence, Union, Any

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import zipfile
import cv2
import numpy as np
import torch
import csv
from pathlib import Path

from datasets import TimbreDataModule
from datasets.timbreutils import *
from experiment_music import MusicVAELightningModule
from utils import get_config, dotdict
import os
import pickle
import re
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import inspect
import soundfile as sf
import math
import sys


class AudioSTFT(Dataset):
    def __init__(
            self,
            config,
    ):
        super().__init__()

        self.dataset_config = config
        # print(f"Here is the loaded config: {self.dataset_config}")

        self.timbre_data = TimbreDataModule(config.timbre)
        self.timbre_data.setup()
        self.timbre_dl = self.timbre_data.train_dataloader()

        chk_path = self.dataset_config.music_vae.checkpoint_path  # os.path.join("/Users/pratik/repos/TimbreSpace",  f"logw/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt")

        self.timbre_model = MusicVAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                                         map_location=torch.device('cpu'),
                                                                         )
        # if self.dataset_config.saver.enabled:  # TODO: untested
        #     print(f"saver enabled")
        #     saver = Saver(self.dataset_path, self.dataset_path)  # Saves spectrograms (.npy) and min/max values
        #     self.preprocessing_pipeline.saver = saver
        #
        # visualize_path = self.dataset_path / self.dataset_config.visualizer.save_dir / 'spectrogram_img'
        # if not visualize_path.exists():
        #     os.makedirs(visualize_path)
        # visualizer = Visualizer(visualize_path, self.dataset_config.stft.frame_size,
        #                         self.dataset_config.stft.hop_length)
        # self.preprocessing_pipeline.visualizer = visualizer

    def __getitem__(self, key):  # Get random segment. key in the param is not used
        batch, batch_di, signal, signal_di, key, offset = next(iter(self.timbre_dl))

        if self.dataset_config.audio == "original":
            return torch.squeeze(batch_di, 0), torch.squeeze(signal_di, 0)
        elif self.dataset_config.audio == "reconstructed":
            di_recons, _, _, _, z = self.timbre_model.forward(torch.squeeze(batch, 0))
            return torch.squeeze(di_recons, 1), torch.squeeze(signal_di, 0)



    def __len__(self):
        return len(self.timbre_data.dataset.clips)


class AudioSTFTDataModule(LightningDataModule, ABC):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        batch_size: the batch size to use.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            config: dict,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.num_workers = config.timbre.num_workers
        self.pin_memory = pin_memory
        self.config = config

    def setup(self) -> None:
        self.dataset = AudioSTFT(
            self.config,
        )

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=1,  # this is not the actual batch size
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader()
