from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datasets import TimbreDataModule

data = TimbreDataModule('../../data/timbre')
data.setup()

dl = data.train_dataloader()
# print(next(iter(dl)))
for batch in iter(dl):
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2])
    print(batch[3])
    exit()