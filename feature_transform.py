#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Dict, OrderedDict, Tuple
from classes import dataclass
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor

from data_loader import DataReader


class DenseFeatureTransform:
    def __init__(
        self,
        data_reader: DataReader,
        is_train: bool = True
    ) -> None:
        pass


class SparseFeatureTransform:
    def __init__(
        self,
        data_reader: DataReader,
        is_train: bool = True
    ) -> None:
        pass
