import os
import io
from abc import ABC
from typing import List, Optional, Sequence, Union, Any

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.celeba import CSV, CelebA
from torch.utils.data import Dataset
import zipfile
import cv2
import numpy as np
import torch
import csv
from pathlib import Path

from datasets.timbreutils import *
from utils import get_config
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


class TimbreDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            cache_into_memory=True,
    ):
        super().__init__()

        self.dataset_path = Path(dataset_path)

        self.config_file = self.dataset_path / 'config.yaml'
        if self.config_file.exists():
            self.config = get_config(self.config_file)
        else:
            raise Exception("Config not found. Dataset path might be incorrect or Dataset might be incorrectly "
                            "configured!")
        print(f"Here is the loaded config: {self.config}")

        if self.config.load.clip_duration is None: # TODO: this is NOT handled
            y, sr = librosa.load(self.dataset_path / "DI.wav", sr=self.config.load.sample_rate, mono=self.config.load.mono)
            self.config.load['clip_duration'] = librosa.get_duration(y=y, sr=sr)
        print(f"Duration of the DI: {self.config.load.clip_duration}")

        self.batch_duration = self.config.data_params.batch_size * self.config.stft.segment_duration
        print(f"Load duration: {self.batch_duration}")

        if self.batch_duration >= self.config.load.clip_duration:
            raise Exception("Batch Duration > Clip Duration")

        self.clips = [name for name in os.listdir(self.dataset_path / 'clips') if name.endswith('.wav')]

        loader = Loader(self.config.load.sample_rate, self.batch_duration, self.config.load.mono)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(self.config.stft.frame_size, self.config.stft.hop_length)
        # feature_extractor = FeatureExtractor()
        min_max_normaliser = MinMaxNormaliser(0, 1)

        self.preprocessing_pipeline = PreprocessingPipeline(self.dataset_path, self.config)
        self.preprocessing_pipeline.loader = loader
        self.preprocessing_pipeline.padder = padder
        self.preprocessing_pipeline.spectrogram_extractor = log_spectrogram_extractor
        # self.preprocessing_pipeline.feature_extractor = feature_extractor
        self.preprocessing_pipeline.normaliser = min_max_normaliser

        saver = Saver(self.dataset_path, self.dataset_path)
        self.preprocessing_pipeline.saver = saver

        visualizer = Visualizer(Path(__file__).parents[1] / 'out', self.config.stft.frame_size, self.config.stft.hop_length)
        self.preprocessing_pipeline.visualizer = visualizer

    def __getitem__(self, key):  # Get random segment
        offset = np.random.uniform(0, self.config.load.clip_duration - self.batch_duration)
        print(f"Getting segment from clip: {key} -> {self.clips[key]}")
        print(f"Offset: {offset}")

        batch, batch_di = self.preprocessing_pipeline.process_file(self.clips[key], offset)
        return batch, batch_di, key, offset

    def __len__(self):
        return len(self.clips)


class TimbreDataModule(LightningDataModule, ABC):
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
            dataset_path: str,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self) -> None:
        self.dataset = TimbreDataset(
            self.dataset_path,
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
