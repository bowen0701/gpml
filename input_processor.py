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
            features_df.loc[:, self.data_reader.float_feature_names]
        ).values
        id_list_examples_np = (
            features_df.loc[:, self.data_reader.id_list_feature_names]
        ).values
        id_score_list_examples_np = (
            features_df.loc[:, self.data_reader.id_score_list_feature_names]
        ).values
        embedding_examples_np = (
            features_df.loc[:, self.data_reader.embedding_feature_names]
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
    ) -> None:
        id_list_feature_metadata = OrderedDict()

        for c in range(id_list_examples_np.shape[1]):
            col = id_list_examples_np[:, c]
            unique_data = np.unique(col)
            data_idx_map = {data: idx for idx, data in enumerate(unique_data)}
            id_list_feature_metadata[
                self.data_reader.id_list_feature_names[c]
            ] = data_idx_map
        
        self.id_list_feature_metadata = id_list_feature_metadata
    
    def preproc_id_list_features(
        self,
        id_list_examples_np: np.ndarray,
    ) -> torch.Tensor:
        id_list_features_preproc_np = deepcopy(id_list_examples_np)

        for c in range(id_list_features_preproc_np.shape[1]):
            # Convert category data to idx, with unknown category mapping to largest idx + 1.
            # Note: The unknown category would only appear in the test data.
            data_idx_map = self.id_list_feature_metadata[
                self.data_reader.id_list_feature_names[c]
            ]
            data2idx = lambda x: data_idx_map.get(x, len(data_idx_map))
            result = np.array(list(map(data2idx, id_list_features_preproc_np[:, c])))
            id_list_features_preproc_np[:, c] = result
        
        id_list_features_preproc = torch.from_numpy(
            id_list_features_preproc_np.astype(np.int64)
        )
        return id_list_features_preproc
