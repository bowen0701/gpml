!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Lambda


class CustomDataset(Dataset):
    def __init__(
        self, 
        data_reader: Any = None, 
        transform: Any = None, 
        target_transform: Any = None
    ) -> None:
        self.examples, self.labels = data_reader()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, float]:
        example = self.examples[idx, :]
        label = self.labels[idx]
        if self.transform:
            example = self.transform(example)
        if self.target_transform:
            label = self.target_transform(self)
        return example, label
