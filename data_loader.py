#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Any, List, Optional, Tuple
from classes import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


@dataclass
class InputData:
    features: np.ndarray
    labels: np.ndarray


class DataReader:
    def __init__(
        self, 
        label_name: str,
        file_name: str, 
        float_feature_names: Optional[List[str]] = None, 
        id_list_feature_names: Optional[List[str]] = None, 
        id_score_list_feature_names: Optional[List[str]] = None, 
        embedding_feature_names: Optional[List[str]] = None, 
    ) -> None:
        self.file_name = file_name
        self.label_name = label_name

        if float_feature_names is None:
            float_feature_names = []
        if id_list_feature_names is None:
            id_list_feature_names = []
        if id_score_list_feature_names is None:
            id_score_list_feature_names = []
        if embedding_feature_names is None:
            embedding_feature_names = []

        self.float_feature_names = float_feature_names
        self.id_list_feature_names = id_list_feature_names
        self.id_score_list_feature_names = id_score_list_feature_names
        self.embedding_feature_names = embedding_feature_names
        self.feature_names = (
            self.float_feature_names + 
            self.id_list_feature_names + 
            self.id_score_list_feature_names + 
            self.embedding_feature_names
        )
        if len(self.feature_names) < 1:
            raise ValueError("Need to at least set up one feature name.")
    
    def __call__(self) -> InputData:
        data_df = pd.read_csv(self.file_name)
        features_df, labels_df = (
            data_df.loc[:, self.feature_names], data_df.loc[:, self.label_name]
        )
        return InputData(
            features=features_df,
            labels=labels_df
        )


class CustomDataset(Dataset):
    def __init__(
        self, 
        data_reader: DataReader, 
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
