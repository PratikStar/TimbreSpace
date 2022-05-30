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
import zipfile
import cv2
import numpy as np
import torch
import csv


# Reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py
# https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
class CelebAZipDataset(VisionDataset):
    def __init__(self, root_path, transform=None, cache_into_memory=True):
        super().__init__(root_path, transform=transform)

        self.root_path = root_path
        zip_file_path = os.path.join(root_path, 'img_align_celeba.zip')
        if cache_into_memory:
            f = open(zip_file_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(zip_file_path, 'r')

        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))

        mask = slice(None)
        attr = self._load_csv(os.path.join(self.root_path, "list_attr_celeba.txt"), header=1)
        self.attr = attr.data[mask]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(filename) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        arr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)

        pil_img = Image.fromarray(arr[:, :, ::-1])  # because the current mode is BGR
        # pil_img.save('savedimage.jpg')
        target = self.attr[key, :]

        if self.transform is not None:
            pil_img = self.transform(pil_img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return pil_img, target

    def __len__(self):
        return len(self.name_list)


class CelebAZipDataModule(LightningDataModule, ABC):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.data_dir = os.path.join(os.getcwd(), data_path, 'celeba')
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(), ])

        self.train_dataset = CelebAZipDataset(
            self.data_dir,
            # split='train',
            transform=train_transforms,
            # download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = CelebAZipDataset(
            self.data_dir,
            # split='test',
            transform=val_transforms,
            # download=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class CelebADataModule(LightningDataModule, ABC):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.data_dir = os.path.join(os.getcwd(), data_path)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(), ])

        self.train_dataset = CelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = CelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
