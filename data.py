#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class DataReader:
    def __init__(
        self, 
        file_name: str, 
        feature_names: List[str], 
        label_name: str
    ) -> None:
        self.file_name = file_name
        self.feature_names = feature_names
        self.label_name = label_name
    
    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_df = pd.read_csv(self.file_name)
        examples_df, labels_df = (
            data_df.loc[:, self.feature_names], data_df.loc[:, self.label_name]
        )
        return examples_df, labels_df


class CustomDataset(Dataset):
    def __init__(
        self, 
        data_reader: Any = None, 
        transform: Any = None, 
        target_transform: Any = None,
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
