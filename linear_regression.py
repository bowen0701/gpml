from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

np.random.seed(71)


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

    def loss(self, y, y_hat):
        """Squared error loss.

        # squared_error_loss(y, y_hat) 
        #   = - 1/n * \sum_{i=1}^n (y_i - y_hat_i)^2
        """
        self.squared_error = np.square(y - y_hat)
        return np.mean(self.squared_error)

    def _optimize(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_hat = self._model(X) 
        dw = 1 / m * np.matmul(X.T, y_hat - y)
        db = np.mean(y_hat - y)

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

    def predict(self, X):
        return self._model(X).reshape((-1,))


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


if __name__ == '__main__':
    main()
