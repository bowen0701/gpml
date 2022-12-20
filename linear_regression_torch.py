# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class LinearRegression(nn.Module):
    """PyTorch implementation of Linear Regression."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        # Linear regression model.
        self.fc1 = nn.Linear(self.in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return x
