from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


class LinearRegression(object):
    """MXNet/Gluon implementation of Linear Regression."""
    def __init__(self, batch_size=10, lr=0.01, num_epochs=5):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def _data_iter(self):
        dataset = gdata.ArrayDataset(self.features, self.labels)
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
        return gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': self.lr})

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

        net = self._linreg()
        self._weights_init(net)
        loss = self._squared_loss()
        trainer = self._sgd_trainer(net)

        for epoch in list(range(self.num_epochs)):
            for X, y in self._data_iter():
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer.step(self.batch_size)

            train_loss = loss(net(self.features), self.labels)
            print('epoch {0}: loss {1}'
                  .format(epoch + 1, train_loss.mean().asnumpy()))

        self.net = net
        return self


def main():
    true_w = nd.array([2, -3.4])
    true_b = 4.2

    num_examples = 1000
    num_input = len(true_w)

    features = nd.random.normal(scale=1, shape=(num_examples, num_input))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    linreg = LinearRegression(batch_size=10, lr=0.02, num_epochs=5)
    linreg.fit(features, labels)

    print('w, true_w: {0}, {1}'.format(linreg.net[0].weight.data(), true_w))
    print('b, true_b: {0}, {1}'.format(linreg.net[0].bias.data(), true_b))


if __name__ == '__main__':
    main()
