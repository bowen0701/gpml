from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

np.random.seed(71)


class PerceptronClassifier(object):
    """Numpy implementation of Perceptron."""

    def __init__(self, batch_size=64, lr=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def get_data(self, X_train, y_train, shuffle=True):
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
        """Perceptron linear regression model.
        
        y = sign(Xw + b), where 
        - y = 1 if Xw + b > 0
        - y = -1 if Xw + b < 0
        """
        self.weighted_sum = np.matmul(X, self.w) + self.b
        return np.sign(self.weighted_sum)

    def _loss(self, y, y_):
        """Hinge loss.

        hinge_loss(y, y_) 
          = 1/n * \sum_{i=1}^n (y_i * (Xw + b)) (y_i != y__i)
        """
        self.hinge_loss = y * self.weighted_sum * self.is_mismatch
        return np.mean(self.hinge_loss)

    def _optimize(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_ = self._model(X)
        self.is_mismatch = np.not_equal(y, y_)
        dw = 1 / m * np.matmul(X.T, self.is_mismatch * y)
        db = np.mean(self.is_mismatch)

        for (param, grad) in zip([self.w, self.b], [dw, db]):
            param[:] = param + self.lr * grad

    def _fetch_batch(self):
        """Fetch batch dataset."""
        idx = list(range(self.n_examples))
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = idx[i:min(i + self.batch_size, self.n_examples)]
            yield (self.X_train.take(idx_batch, axis=0), 
                   self.y_train.take(idx_batch, axis=0))

    def fit(self):
        """Fit model."""
        self._create_weights()

        for epoch in range(self.n_epochs):
            total_loss = 0
            for X_train_b, y_train_b in self._fetch_batch():
                y_train_b = y_train_b.reshape((y_train_b.shape[0], -1))
                self._optimize(X_train_b, y_train_b)
                batch_loss = self._loss(y_train_b, self._model(X_train_b))
                total_loss += batch_loss * X_train_b.shape[0]

            if epoch % 100 == 0:
                print('epoch {0}: training loss {1}'
                      .format(epoch, total_loss / self.n_examples))

        return self

    def get_coeff(self):
        """Get model coefficients."""
        return self.b, self.w.reshape((-1,))

    def predict(self, X):
        """Predict for new data."""
        return self._model(X).reshape((-1,))


def main():
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import Perceptron as PerceptronSklearn

    import sys
    sys.path.append('../numpy/')
    from metrics import accuracy

    # Read breast cancer data.
    X, y = load_breast_cancer(return_X_y=True)
    y = y * 2 - 1

    # Split data into training and test datasets.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=71, shuffle=True)

    # Feature engineering for standardizing features by min-max scaler.
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_raw)
    X_test = min_max_scaler.transform(X_test_raw)

    # Convert arrays to float32.
    X_train, X_test, y_train, y_test = (
        np.float32(X_train), np.float32(X_test), 
        np.float32(y_train), np.float32(y_test)
    )

    # Fit Perceptron Classfier.
    perceptron = PerceptronClassifier(batch_size=64, lr=0.01, n_epochs=1000)
    # Get datasets and build graph.
    perceptron.get_data(X_train, y_train, shuffle=True)
    perceptron.fit()

    print(perceptron.get_coeff())
    # Predicted probabilities for training data.
    y_train_ = perceptron.predict(X_train)
    print('Training accuracy: {}'
           .format(accuracy(y_train, y_train_)))
    y_test_ = perceptron.predict(X_test)
    print('Test accuracy: {}'
           .format(accuracy(y_test, y_test_)))

    # Benchmark with Sklearn's Perceptron.
    print('Train Sklearn Perceptron:')
    perceptron_sk = PerceptronSklearn(max_iter=1000)
    perceptron_sk.fit(X_train, y_train.reshape(y_train.shape[0], ))

    y_train_ = perceptron_sk.predict(X_train)
    print('Training accuracy: {}'.format(accuracy(y_train, y_train_)))
    y_test_ = perceptron_sk.predict(X_test)
    print('Test accuracy: {}'.format(accuracy(y_test, y_test_)))


if __name__ == '__main__':
    main()
