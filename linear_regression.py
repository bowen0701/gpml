from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import tensorflow as tf
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


class LinearRegression(object):
    """Numpy implementation of Linear Regression."""
    def __init__(self, batch_size=64, lr=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def get_dataset(self, X_train, y_train, shuffle=True):
        """Get dataset and information."""
        self.X_train = X_train
        self.y_train = y_train

        # Get the numbers of examples and inputs.
        self.n_examples, self.n_inputs = self.X_train.shape

        if shuffle:
            idx = list(range(self.n_examples))
            random.shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]

    def _create_weights(self):
        """Create model weights and bias."""
        self.w = np.zeros(self.n_inputs).reshape(self.n_inputs, 1)
        self.b = np.zeros(1).reshape(1, 1)

    def _model(self, X):
        """Linear regression model."""
        return np.matmul(X, self.w) + self.b

    def loss(self, y, logit):
        """Cross entropy loss by stabilizaiton trick.

        # cross_entropy_loss(y, z) 
        #   = - 1/n * \sum_{i=1}^n y_i * log p(y_i = 1|x_i) + (1 - y_i) * log p(y_i = 0|x_i)
        #   = - 1/n * \sum_{i=1}^n y_i * (z_i - log(1 + exp(z_i))) + (1 - y_i) * (-log(1 + exp(z_i))),
        # where z is the logit, z_max = z - max(0, z),
        #   log p(y = 1|x)
        #     = log (1 / (1 + exp(-z))) 
        #     = log (exp(z) / (1 + exp(z)))
        #     = z - log(1 + exp(z))
        # and 
        #   log(1 + exp(z)) := logsumexp(z)
        #     = log(exp(0) + exp(z))
        #     = log(exp(0) + exp(z) * exp(z_max) / exp(z_max))
        #     = z_max + log(exp(-z_max) + exp(z - z_max)).
        """
        # logit_max = np.maximum(0, logit)
        # logit_stable = logit - logit_max
        # logsumexp_stable = logit_max + np.log(np.exp(-logit_max) + np.exp(logit_stable))
        # self.cross_entropy = -(y * (logit - logsumexp_stable) + (1 - y) * (-logsumexp_stable))
        # return np.mean(self.cross_entropy)

    def _optimize(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_hat = self._model(X) 
        dw = -1 / m * np.matmul(X.T, y - y_hat)
        db = -np.mean(y - y_hat)

        for (param, grad) in zip([self.w, self.b], [dw, db]):
            param[:] = param - self.lr * grad

    def _fetch_batch(self):
        """Fetch batch dataset."""
        idx = list(range(self.n_examples))
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = idx[i:min(i + self.batch_size, self.n_examples)]
            yield (self.X_train.take(idx_batch, axis=0), self.y_train.take(idx_batch, axis=0))

    def fit(self):
        """Fit model."""
        # self._create_weights()

        # for epoch in range(self.n_epochs):
        #     total_loss = 0
        #     for X_train_b, y_train_b in self._fetch_batch():
        #         y_train_b = y_train_b.reshape((y_train_b.shape[0], -1))
        #         self._optimize(X_train_b, y_train_b)
        #         train_loss = self.loss(y_train_b, self.logit(X_train_b))
        #         total_loss += train_loss * X_train_b.shape[0]

        #     if epoch % 100 == 0:
        #         print('epoch {0}: training loss {1}'.format(epoch, total_loss))

        # return self
        pass

    def get_coeff(self):
        return self.b, self.w.reshape((-1,))

    def predict(self, X_test):
        return self._model(X_test).reshape((-1,))


def reset_tf_graph(seed=71):
    """Reset default TensorFlow graph."""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class LinearRegressionTF(object):
    """A TensorFlow implementation of Linear Regression."""
    def __init__(self, batch_size=64, learning_rate=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def get_dataset(self, X_train, y_train, shuffle=True):
        """Get dataset and information.s"""
        self.X_train = X_train
        self.y_train = y_train

        # Get the numbers of examples and inputs.
        self.n_examples, self.n_inputs = self.X_train.shape

        idx = list(range(self.n_examples))
        if shuffle:
            random.shuffle(idx)
        self.X_train = self.X_train[idx]
        self.y_train = self.y_train[idx]
    
    def _create_placeholders(self):
        """Create placeholder for features and response."""
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    
    def _create_weights(self):
        """Create and initialize model weights and bias."""
        self.w = tf.get_variable(shape=(self.n_inputs, 1), 
                                 initializer=tf.random_normal_initializer(0, 0.01), 
                                 name='weights')
        self.b = tf.get_variable(shape=(1, 1), 
                                 initializer=tf.zeros_initializer(), name='bias')
    
    def _create_model(self):
        """Create linear model."""
        self.y_pred = tf.add(tf.matmul(self.X, self.w), self.b, name='y_pred')
    
    def _create_loss(self):
        # Create mean squared error loss.
        self.error = self.y_pred - self.y
        self.loss = tf.reduce_mean(tf.square(self.error), name='loss')
    
    def _createoptimizer(self):
        # Create gradient descent optimization.
        self.optimizer = (
            tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            .minimize(self.loss))

    def build_graph(self):
        """Build computational graph."""
        self._create_placeholders()
        self._create_weights()
        self._create_model()
        self._create_loss()
        self._create_optimizer()

    def _fetch_batch(self):
        """Fetch batch dataset."""
        idx = list(range(self.n_examples))
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = idx[i:min(i + self.batch_size, self.n_examples)]
            yield (self.X_train[idx_batch, :], self.y_train[idx_batch, :])

    def fit(self):
        """Fit model."""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.n_epochs):
                total_loss = 0

                for X_train_b, y_train_b in self._fetch_batch():
                    feed_dict = {self.X: X_train_b, self.y: y_train_b}
                    _, batch_loss = sess.run([self.optimizer, self.loss],
                                             feed_dict=feed_dict)
                    total_loss += batch_loss

                if epoch % 100 == 0:
                    print('Epoch {0}: training loss: {1}'
                          .format(epoch, total_loss / self.n_examples))


class LinearRegressionMX(object):
    """MXNet implementation of Linear Regression."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def get_dataset(self, X_train, y_train, shuffle=True):
        """Get dataset and information.s"""
        self.X_train = X_train
        self.y_train = y_train

        # Get the numbers of examples and inputs.
        self.n_examples, self.n_inputs = self.X_train.shape

        idx = list(range(self.n_examples))
        if shuffle:
            random.shuffle(idx)
        self.X_train = self.X_train[idx]
        self.y_train = self.y_train[idx]
    
    def _linreg(self, X, w, b):
        """Linear model."""
        return nd.dot(X, w) + b
    
    def _squared_loss(self, y_hat, y):
        """Squared loss."""
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def _create_weights(self):
        """Create and initialize model weights and bias and attach gradients."""
        w = nd.random.normal(scale=0.01, shape=(self.n_inputs, 1))
        b = nd.zeros(shape=(1,))
        params = [w, b]
        for param in params:
            # Attach gradient for automatic differentiation.
            param.attach_grad()
        return params
    
    def _sgd(self, w, d):
        """Apply stochastic gradient descent."""
        for param in [w, d]:
            # Take parameter's gradient from auto diff output.
            param[:] = param - self.lr * param.grad / self.batch_size

    def _fetch_batch(self):
        idx = list(range(self.n_examples))
        random.shuffle(idx)
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = nd.array(idx[i:min(i + self.batch_size, self.n_examples)])
            yield self.X_train.take(idx_batch), self.y_train.take(idx_batch)
    
    def fit(self, X_train, y_train):
        net = self._linreg
        loss = self._squared_loss
        w, b = self._create_weights()

        for epoch in range(self.n_epochs):
            total_loss = 0

            for X_train_b, y_train_b in self._data_iter():
                # Record auto diff & perform backward differention.
                with autograd.record():
                    l = loss(net(X_train_b, w, b), y_train_b)
                l.backward()
                self._sgd(w, b)

                # TODO: Refactor fit()'s batch loss and total loss.
                # batch_loss = loss(net(self.X_train, w, b), self.y_train)
                # total_loss += batch_loss

            # train_loss = loss(net(self.X_train), self.y_train)
            # print('epoch {0}: loss {1}'
            #       .format(epoch + 1, train_loss.mean().asnumpy()))

            # if epoch % 100 == 0:
            #     print('Epoch {0}: training loss: {1}'
            #           .format(epoch, train_loss.mean().asnumpy()))

        self.net = net
        self.w, self.b = w, b
        return self

    def get_coeff(self):
        return self.b, self.w.reshape((-1,))

    def predict(self, X_test):
        return self.net(X_test, self.w, self.b).reshape((-1,))


class LinearRegressionMXGluon(object):
    """MXNet-Gluon implementation of Linear Regression."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def _linreg(self):
        """Linear model."""
        net = nn.Sequential()
        net.add(nn.Dense(1))
        return net

    def _squared_loss(self):
        """Squared loss."""
        return gloss.L2Loss()

    def _create_weights(self, net):
        """Create and initialize model weights and bias."""
        net.initialize(init.Normal(sigma=0.01))

    def _sgd_trainer(self, net):
        """Stochastic gradient descent trainer."""
        return gluon.Trainer(
            net.collect_params(), 'sgd', {'learning_rate': self.lr})

    def _fetch_batch(self):
        dataset = gdata.ArrayDataset(self.X_train, self.y_train)
        return gdata.DataLoader(dataset, self.batch_size, shuffle=True)

    def fit(self, X_train, y_train):
        net = self._linreg()
        loss = self._squared_loss()
        self._create_weights(net)
        trainer = self._sgd_trainer(net)

        for epoch in list(range(self.n_epochs)):
            for X, y in self._fetch_batch():
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer.step(self.batch_size)

                # TODO: Refactor fit()'s batch loss and total loss.
                # batch_loss = loss(net(self.X_train, w, b), self.y_train)
                # total_loss += batch_loss

            train_loss = loss(net(self.X_train), self.y_train)
            print('epoch {0}: loss {1}'
                  .format(epoch + 1, train_loss.mean().asnumpy()))

            # if epoch % 100 == 0:
            #     print('Epoch {0}: training loss: {1}'
            #           .format(epoch, train_loss.mean().asnumpy()))

        self.net = net
        return self

    def get_coeff(self):
        _coef = self.net[0]
        return _coef.bias.data(), _coef.weight.data().reshape((-1,))

    def predict(self, X_test):
        return self.net(X_test).reshape((-1,))


def main():
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    # Read California housing data.
    housing = fetch_california_housing()
    data = housing.data
    label = housing.target.reshape(-1, 1)

    # Normalize features first.
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split data into training/test datasets.
    test_ratio = 0.2
    test_size = int(data.shape[0] * test_ratio)

    X_train = data[:-test_size]
    X_test = data[-test_size:]
    y_train = label[:-test_size]
    y_test = label[-test_size:]

    # TODO: Train Numpy linear regression model.

    # Train TensorFlow linear regression model.
    reset_tf_graph()
    linreg = LinearRegressionTF()
    linreg.get_dataset(X_train, y_train)
    linreg.build_graph()
    linreg.fit()

    # TODO: Train MXNet linear regression model.

    # TODO: Train MXNet-Gluon linear regression model.


if __name__ == '__main__':
    main()
