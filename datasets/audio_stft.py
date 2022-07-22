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
            dataset_path,
            config,
            cache_into_memory=True,
    ):
        super().__init__()

        self.dataset_path = Path(dataset_path)

        self.dataset_config_file = self.dataset_path / 'config.yaml'
        if self.dataset_config_file.exists():
            self.dataset_config = get_config(self.dataset_config_file)
        else:
            raise Exception("Config not found. Dataset path might be incorrect or Dataset might be incorrectly "
                            "configured!")

        self.dataset_config = dotdict({**config, **self.dataset_config})
        # print(f"Here is the loaded config: {self.dataset_config}")

        if self.dataset_config.load.clip_duration is None: # TODO: this is NOT handled
            y, sr = librosa.load(self.dataset_path / "DI.wav", sr=self.dataset_config.load.sample_rate, mono=self.dataset_config.load.mono)
            self.dataset_config.load['clip_duration'] = librosa.get_duration(y=y, sr=sr)
        # print(f"Duration of the DI: {self.dataset_config.load.clip_duration}")

        self.batch_duration = self.dataset_config.batch_size * self.dataset_config.stft.segment_duration
        # print(f"Load duration: {self.batch_duration}")

        if self.batch_duration >= self.dataset_config.load.clip_duration:
            raise Exception("Batch Duration > Clip Duration")

        self.clips = [name for name in os.listdir(self.dataset_path / 'clips') if name.endswith('.wav')]

        loader = Loader(self.dataset_config.load.sample_rate, self.batch_duration, self.dataset_config.load.mono)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(self.dataset_config.stft.frame_size, self.dataset_config.stft.hop_length)
        # feature_extractor = FeatureExtractor()
        min_max_normaliser = MinMaxNormaliser(0, 1)

        self.preprocessing_pipeline = PreprocessingPipeline(self.dataset_path, self.dataset_config)
        self.preprocessing_pipeline.loader = loader
        self.preprocessing_pipeline.padder = padder
        self.preprocessing_pipeline.spectrogram_extractor = log_spectrogram_extractor
        # self.preprocessing_pipeline.feature_extractor = feature_extractor
        self.preprocessing_pipeline.normaliser = min_max_normaliser

        chk_path = self.dataset_config.music_vae.checkpoint_path # os.path.join("/Users/pratik/repos/TimbreSpace",  f"logw/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt")

        # chkpt = torch.load(chk_path, map_location=torch.device('cpu'))
        self.model = MusicVAELightningModule.load_from_checkpoint(checkpoint_path=chk_path,
                                                             map_location=torch.device('cpu'),
                                                             )
        if self.dataset_config.saver.enabled: # TODO: untested
            print(f"saver enabled")
            saver = Saver(self.dataset_path, self.dataset_path) # Saves spectrograms (.npy) and min/max values
            self.preprocessing_pipeline.saver = saver

        visualize_path = self.dataset_path / self.dataset_config.visualizer.save_dir / 'spectrogram_img'
        if not visualize_path.exists():
            os.makedirs(visualize_path)
        visualizer = Visualizer(visualize_path, self.dataset_config.stft.frame_size, self.dataset_config.stft.hop_length)
        self.preprocessing_pipeline.visualizer = visualizer

    def __getitem__(self, key):  # Get random segment
        offset = np.random.uniform(0, self.dataset_config.load.clip_duration - self.batch_duration) # 84.4577468515507 this offset was used to generating reconstruction spectrograms
        # print(f"Getting segment from clip: {key} -> {self.clips[key]}")
        # print(f"Offset: {offset}")

        batch, batch_di = self.preprocessing_pipeline.process_file(self.clips[key], offset, self.dataset_config.visualizer.enabled)
        # print(batch.shape)
        file_name = self.dataset_path / 'clips' / self.clips[key]
        signal = self.preprocessing_pipeline.loader.load(file_name, offset)
        s = []
        signal_segment_width = signal.shape[0] // self.dataset_config.batch_size
        for i in range(0, self.dataset_config.batch_size):
            s.append(signal[i*signal_segment_width: (i+1) *signal_segment_width])
        di_recons, _, _, _, z = self.model.forward(torch.tensor(batch))

        return di_recons, np.array(s), batch, batch_di, self.clips[key], key, offset

    def __len__(self):
        return len(self.clips)


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

        self.dataset_path = Path(config.dataset_path)
        self.num_workers = config.num_workers
        self.pin_memory = pin_memory
        self.config = config

    def setup(self) -> None:
        self.dataset = AudioSTFT(
            self.dataset_path,
            self.config,
        )

    def get_dataloader(self) -> DataLoader:

        return DataLoader(
            self.dataset,
            batch_size=1, # this is not the actual batch size
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader()
