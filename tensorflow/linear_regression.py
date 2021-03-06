from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

# TensorFlow import.
import tensorflow as tf

np.random.seed(71)


def reset_tf_graph(seed=71):
    """Reset default TensorFlow graph."""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class LinearRegression(object):
    """A TensorFlow implementation of Linear Regression."""

    def __init__(self, batch_size=64, learning_rate=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def get_data(self, X_train, y_train, shuffle=True):
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
        self.w = tf.get_variable(shape=[self.n_inputs, 1],
                                 initializer=tf.random_normal_initializer(),
                                 name='weights')
        self.b = tf.get_variable(shape=[1],
                                 initializer=tf.zeros_initializer(),
                                 name='bias')

    def _model(self, X):
        """Linear regression model."""
        return tf.matmul(X, self.w) + self.b

    def _create_model(self):
        """Create linear model."""
        self.y_ = self._model(self.X)

    def _create_loss(self):
        # Create mean squared error loss.
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y), name='loss')

    def _create_optimizer(self):
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
            yield (self.X_train[idx_batch, :], self.y_train[idx_batch].reshape(-1, 1))

    def fit(self):
        """Fit model."""
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(1, self.n_epochs + 1):
                total_loss = 0
                for X_train_b, y_train_b in self._fetch_batch():
                    feed_dict = {self.X: X_train_b, self.y: y_train_b}
                    _, batch_loss = sess.run([self.optimizer, self.loss],
                                             feed_dict=feed_dict)
                    total_loss += batch_loss * X_train_b.shape[0]

                if epoch % 100 == 0:
                    print('Epoch {0}: training loss: {1}'
                          .format(epoch, total_loss / self.n_examples))

            # Save model.
            saver.save(sess, 'checkpoints/linreg')

    def get_coeff(self):
        """Get model coefficients."""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load model.
            saver = tf.train.Saver()
            saver.restore(sess, 'checkpoints/linreg')
            return self.b.eval(), self.w.eval().reshape((-1,))

    def predict(self, X):
        """Predict for new data."""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load model.
            saver = tf.train.Saver()
            saver.restore(sess, 'checkpoints/linreg')
            return self._model(X).eval().reshape((-1,))


def main():
    import sklearn
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression as LinearRegressionSklearn
    from metrics import mean_squared_error

    # Read California housing data.
    housing = fetch_california_housing()
    data = housing.data
    label = housing.target

    # Normalize features first.
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

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

    # Train TensorFlow linear regression model.
    linreg_tf = LinearRegression(batch_size=64, learning_rate=0.1, n_epochs=1000)
    linreg_tf.get_data(X_train, y_train, shuffle=True)
    linreg_tf.fit()
    print(linreg_tf.get_coeff())
    y_train_ = linreg_tf.predict(X_train)
    print('Training mean squared error: {}'
           .format(mean_squared_error(y_train, y_train_)))
    y_test_ = linreg_tf.predict(X_test)
    print('Test mean squared error: {}'
           .format(mean_squared_error(y_test, y_test_)))

    # Benchmark with sklearn's linear regression model.
    linreg_sk = LinearRegressionSklearn()
    linreg_sk.fit(X_train, y_train) 
    y_train_ = linreg_sk.predict(X_train)
    print('Training mean squared error: {}'
           .format(mean_squared_error(y_train, y_train_)))
    y_test_ = linreg_sk.predict(X_test)
    print('Test mean squared error: {}'
           .format(mean_squared_error(y_test, y_test_)))


if __name__ == '__main__':
    main()
