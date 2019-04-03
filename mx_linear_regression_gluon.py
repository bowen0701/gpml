from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


class LinearRegression(object):
    """MXNet/Gluon implementation of Linear Regression."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def _data_iter(self):
        dataset = gdata.ArrayDataset(self.X_train, self.y_train)
        return gdata.DataLoader(dataset, self.batch_size, shuffle=True)

    def _linreg(self):
        net = nn.Sequential()
        net.add(nn.Dense(1))
        return net

    def _weights_init(self, net):
        net.initialize(init.Normal(sigma=0.01))

    def _squared_loss(self):
        return gloss.L2Loss()

    def _sgd_trainer(self, net):
        return gluon.Trainer(
            net.collect_params(), 'sgd', {'learning_rate': self.lr})

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        net = self._linreg()
        self._weights_init(net)
        loss = self._squared_loss()
        trainer = self._sgd_trainer(net)

        for epoch in list(range(self.n_epochs)):
            for X, y in self._data_iter():
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer.step(self.batch_size)

            train_loss = loss(net(self.X_train), self.y_train)
            print('epoch {0}: loss {1}'
                  .format(epoch + 1, train_loss.mean().asnumpy()))

        self.net = net
        return self

    def coef(self):
        _coef = self.net[0]
        return _coef.bias.data(), _coef.weight.data()

    def predict(self, X_test):
        return self.net(X_test)
