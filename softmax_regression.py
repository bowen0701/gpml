from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autograd
import random


class SoftmaxRegressionMX(object):
    """MXNet implementation of Softmax Regression."""
    def __init__(self, batch_size=256, lr=0.01, n_epoches=5):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epoches = n_epoches

    def _softmax(self, X):
        X_exp = X.exp()
        partition = X_exp.sum(axis=1, keepdims=True)
        return X_exp / partition

    def _softmaxreg(self, X, w, b):
        return self._softmax(nd.dot(X.reshape((-1, self.n_inputs)), w) + b)

    def _cross_entropy(self, y_hat, y):
        return -nd.log(nd.pick(y_hat, y))

    def _accuracy(self, y_hat, y):
        return ((y_hat.argmax(axis=1).astype('float32') == y.astype('float32'))
                .mean()
                .asscalar())

    def _eval_accuracy(self, X, y):
        y_hat = self.net(self.X, self.w, self.b)
        return self._accuracy(y_hat, y)

    def _weights_init(self):
        w = nd.random.normal(scale=0.01, shape=(self.n_inputs, self.n_outputs))
        b = nd.zeros(shape=(self.n_outputs,))
        params = [w, d]
        for param in params:
            param.attach_grad()
        return params

    def _sgd(self, w, d):
        for param in [w, d]:
            param[:] = param - self.lr * param.grad / self.batch_size

    def _data_iter(self):
        idx = list(range(self.n_examples))
        random.shuffle(idx)
        for i in range(0, n_examples, self.batch_size):
            idx_batch = nd.array(idx[i:min(i + self.batch_size, self.n_examples)])
            yield self.X_train.take(idx_batch), self.y_train.take(idx_batch)

    def _get_n_inputs(self, X_train):
        example = X_train[0]
        example_shape = example.reshape(example.shape[0], -1).shape
        return example_shape[0] * example_shape[1]

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_examples = len(X_train)
        self.n_inputs = self._get_n_inputs(X_train)

        net = self._softmaxreg
        loss = self._cross_entropy
        w, b = self._weights_init()

        for epoch in range(self.n_epoches):
            for X, y in self._data_iter():
                with autograd.record():
                    l = loss.(net(X, w, b))
                l.backward()
                self._sgd(w, d)

            train_loss = loss(net(self.X_train, w, b), self.y_train)
            train_accuracy = self._eval_accuracy(self.X_train, self.y)
            print('epoch {0}: loss {1}, accuracy {2}'
                  .format(epoch + 1,
                          train_loss.mean().asnumpy(),
                          train_accuracy.mean().asnumpy()))

        self.net = net
        self.w, self.b = w, b
        return self

    def get_coeff(self):
        return self.b, self.w.reshape((-1,))

    def predict(self, X_test):
        return self.net(X_test, self.w, self.b).reshape((-1,))


# TODO: Implement main() with tests.
