from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np


class LogisticRegression(object):
    """Numpy implementation of Logistic Regression."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        self._batch_size = batch_size
        self._lr = lr
        self._n_epochs = n_epochs

    def _sigmoid(self, z):
        def f(x):
            if x < 0:
                # To handle underflow.
                return np.exp(x) / (1 + np.exp(x))
            else:
                # To handle overflow.
                return 1 / (1 + np.exp(-x))
        return np.array(list(map(f, z)))

    def _logreg(self, X, w, b):
        return self._sigmoid(np.dot(X, w) + b)

    def _cross_entropy(self, y_hat, y):
        # TODO: Resolve dividing by zero encountered in log error.
        m = y.shape[0]
        return - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def _weights_init(self):
        w = np.zeros(self._n_inputs).reshape(self._n_inputs, 1)
        b = np.zeros(1).reshape(1, 1)
        return w, b

    def _sgd(self, X, y, w, b):
        m = X.shape[0]

        y_hat = self._logreg(X, w, b) 
        dw = - 1 / m * np.matmul(X.T, y - y_hat)
        db = - np.mean(y - y_hat)
        
        for (param, grad) in zip([w, b], [dw, db]):
            param[:] = param - self._lr * grad

    def _data_iter(self):
        idx = list(range(self._n_examples))
        random.shuffle(idx)
        for i in range(0, self._n_examples, self._batch_size):
            idx_batch = np.array(
                idx[i:min(i + self._batch_size, self._n_examples)])
            yield (self._X_train.take(idx_batch, axis=0), 
                   self._y_train.take(idx_batch, axis=0))

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        self._n_examples, self._n_inputs = X_train.shape

        logreg = self._logreg
        loss = self._cross_entropy
        w, b = self._weights_init()

        for epoch in range(self._n_epochs):
            for step, (X, y) in enumerate(self._data_iter()):
                y = y.reshape((y.shape[0], -1))
                self._sgd(X, y, w, b)
            train_loss = loss(logreg(X, w, b), y)
            if epoch % 10 == 0:
                print('epoch {0}: loss {1}'.format(epoch + 1, train_loss))

        self._logreg = logreg
        self._w, self._b = w, b
        return self

    def get_coeff(self):
        return self._b, self._w.flatten()

    def predict(self, X_test):
        return self._logreg(X_test, self._w, self._b)
