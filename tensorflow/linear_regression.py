from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LinearRegression(object):
    """A TensorFlow implementation of Linear Regression."""
    def __init__(self, batch_size=10, learning_rate=0.01, n_epochs=100):
        self._batch_size = batch_size
        self._n_epochs = n_epocs
        self._learning_rate = learning_rate
    
    def _create_placeholders(self):
        self._X = tf.placeholder(tf.float32, shape=(self._batch_size, self._n_inputs), name='X')
        self._y = tf.placeholder(tf.float32, shape=(self._batch_size, 1), name='y')
    
    def _create_weights(self):
        self._w = tf.get_variable(shape=(self._n_inputs, 1), 
                                  initializer=tf.random_normal_initializer(), 
                                  weights='weights')
        self._b = tf.get_variable(shape=(1, 1),
        	                      initializer=tf.zero_initializer(),
        	                      name='bias')
    
    def _create_model(self):
        self._y_pred = tf.add(tf.matmul(X, w), b, name='y_pred')
    
    def _create_loss(self):
        # Mean squared error as loss.
        self._loss = tf.reduce_mean(tf.square(tf.add(y_pred, -y)), name='loss')
    
    def _create_optimizer(self):
        self._optimizer = (
            tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            .minimize(self._loss))
    
    def _train(self):
        pass

