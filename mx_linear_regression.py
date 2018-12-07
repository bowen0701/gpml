from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import nd, autograd
import random


class LinearRegression(object):
    """A MXNet implementation of Linear Regression."""
    def __init__(self, batch_size=10, lr=0.01, num_epochs=5):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def data_iter(self):
        num_examples = self.features.shape[0]
        idx = list(range(num_examples))
        random.shuffle(idx)
        for i in range(0, num_examples, self.batch_size):
            idx_batch = nd.array(idx[i:min(i + self.batch_size, self.num_examples)])
            yield (nd.take(self.features, idx_batch), 
                   nd.take(self.labels, idx_batch))
    
    def weights_init(self):
        w = nd.random.normal(scale=0.01, shape=(self.num_inputs, 1))
        b = nd.zeros(shape=(1,))
        params = [w, b]
        for param in params:
            # Attach gradient for automatic differentiation.
            param.attach_grad()
        return params
    
    def linreg(self, X, w, b):
        return nd.dot(X, w) + b
    
    def squared_loss(self, y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / (2 * self.batch_size)
    
    def sgd(self, w, d):
        for param in [w, d]:
            # Take parameter's gradient from auto diff output.
            param[:] = param - self.lr * param.grad
    
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_examples = features.shape[0]
        self.num_inputs = features.shape[1]
        
        w, b = self.weights_init()
        
        net = self.linreg
        loss = self.squared_loss

        for epoch in range(self.num_epochs):
            for X, y in self.data_iter():
                with autograd.record():
                    # Record auto diff & perform backward differention.
                    l = loss(net(X, w, b), y)
                    l.backward()
                self.sgd(w, b)

            train_loss = loss(net(self.features, w, b), labels)
            print('epoch {0}: loss {1}'
                  .format(epoch + 1, nd.sum(train_loss).asnumpy()))
        
        self.w, self.b = w, b
        return self


def main():
    true_w = nd.array([2, -3.4])
    true_b = 4.2

    num_examples = 1000
    num_inputs = len(true_w)

    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    linreg = LinearRegression(batch_size=10, lr=0.02, num_epochs=5)

    linreg.fit(features, labels)
    print('w, true_w: {}, {}'.format(linreg.w, true_w))
    print('b, true_b: {}, {}'.format(linreg.b, true_b))


if __name__ == '__main__':
    main()
