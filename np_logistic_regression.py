from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class LogisticRegression(object):
    """Numpy implementation of Logistic Regression."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        self._batch_size = batch_size
        self._lr = lr
        self._n_epochs = n_epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_grad(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _logreg(self, X, w, b):
        return self._sigmoid(np.dot(w.T, X) + b)

    def _cross_entropy(self, y_hat, y): 
        return -1 * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def _weights_init(self):
        w = np.zeros(self._n_inputs).reshape(self._n_inputs, 1)
        b = 0.0
        return w, b

    def _sgd(self, X, y, w, b):
        # TODO: Revise SGD method.
        m = X.shape[0]
        y_hat = self._logreg(X, w, b) 
        dw = 1 / m * np.dot(X, (y_hat - y).T)
        db = 1 / m * np.sum(A - self._y_train)
        
        for param in [w, b]:
            param[:] = param - self.lr * 

    def _data_iter(self):
        idx = list(range(self._n_examples))
        random.shuffle(idx)
        for i in range(0, self._n_examples, self._batch_size):
            idx_batch = np.array(idx[i:min(i + self._batch_size, self._n_examples)])
            yield self._X_train.take(idx_batch), self.y_train.take(idx_batch)

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        self._n_examples, self._n_inputs = X_train.shape

        logreg = self._logreg
        loss = self._cross_entropy
        w, b = self._weights_init()

        for epoch in range(n_epochs):
            for X, y in self._data_iter():
                y_hat = self._logreg(X, w, b)
                loss = self._cross_entropy(y_hat, y)
        pass

    def get_coeff(self):
        pass

    def predict(self, X_test):
        pass
