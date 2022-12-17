from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearRegression(nn.Module):
    """TODO: PyTorch implementation of Linear Regression."""
