from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LinearRegression(object):
    """A TensorFlow implementation of Linear Regression."""
    def __init__(self, batch_size=50, lr=0.01, n_epoch=5):
    	self._batch_size = batch_size
    	self._lr = lr
    	self._n_epoch = n_epoch

    def _linreg(self, X, w, b):
    	return tf.matmul(X, w) + b

    def _squared_lost(self, y_, y):
    	return tf.math.square(y_ - y) / 2

