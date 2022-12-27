#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Dict, OrderedDict, Tuple
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor


class InputProc:
    def __init__(
        self,
        input_data: Dict[str, pd.DataFrame],
        data_reader: DataReader,
    ) -> None:
        self.input_data = input_data
        self.data_reader = data_reader

    def get_feature_groups_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features_df = self.input_data["features"]
        float_examples_np = (
            features_df.loc[:, data_reader.float_feature_names]
        ).values
        id_list_examples_np = (
            features_df.loc[:, data_reader.id_list_feature_names]
        ).values
        id_score_list_examples_np = (
            features_df.loc[:, data_reader.id_score_list_feature_names]
        ).values
        embedding_examples_np = (
            features_df.loc[:, data_reader.embedding_feature_names]
        ).values
        return (
            float_examples_np,
            id_list_examples_np,
            id_score_list_examples_np,
            embedding_examples_np,
        )

    def preproc_id_list_feature_metadata(
        self,
        id_list_examples_np: np.ndarray,
        data_reader: DataReader,
    ) -> OrderedDict[str, Dict[str, int]]:
        id_list_feature_metadata = OrderedDict()

        for c in range(id_list_examples_np.shape[1]):
            col = id_list_examples_np[:, c]
            unique_data = np.unique(col)
            data_idx_map = {data: idx for idx, data in enumerate(unique_data)}
            id_list_feature_metadata[data_reader.id_list_feature_names[c]] = data_idx_map
        return id_list_feature_metadata
