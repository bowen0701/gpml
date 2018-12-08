from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autogrd, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


class LinearRegression(object):
    def __init__(self, batch_size=10, lr=0.01, num_epochs=5):
        pass

    def data_iter(self):
        pass

    def linreg(self):
        pass

    def weights_init(self):
        pass

    def squared_loss(self):
        pass

    def trainer_sgd(self):
        pass

    def fit(self, features, labels):
        pass


def main():
    true_w = nd.array([2, -3.4])
    true_b = 4.2

    num_examples = 1000
    num_input = len(true_w)

    features = nd.random.normal(scale=1, shape=(num_examples, num_input))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    pass


if __name__ == '__main__':
    main()
