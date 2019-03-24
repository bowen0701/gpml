from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LinearRegression(object):
    """A TensorFlow implementation of Linear Regression."""
    pass


def main():
    true_w = nd.array([2, -3.4])
    true_b = 4.2

    num_examples = 1000
    num_inputs = len(true_w)

    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)


if __name__ == '__main__':
    main()
