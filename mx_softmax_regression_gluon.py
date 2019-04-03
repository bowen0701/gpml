from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autograd
import random


class SoftmaxRegression(object):
	"""MXNet implementation of Softmax Regression."""
	def __init__(self, batch_size=10, lr=0.01, n_epoches):
		self.batch_size = batch_size
		self.lr = lr
		self.n_epoches = n_epoches
