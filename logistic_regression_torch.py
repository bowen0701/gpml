# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    """TODO: PyTorch implementation of Linear Regression."""
