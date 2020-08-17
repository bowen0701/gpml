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

    def predict(self, X):
        return self._model(X).reshape((-1,))


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
        self.y_hat = tf.add(tf.matmul(self.X, self.w), self.b, name='y_pred')
    
    def _create_loss(self):
        # Create mean squared error loss.
        self.error = self.y_hat - self.y
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
