from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf


# Reset default graph.
def reset_graph(seed=71):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class LinearRegression(object):
    """A TensorFlow implementation of Linear Regression."""
    def __init__(self, batch_size=64, learning_rate=0.01, n_epochs=1000):
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate

    def get_dataset(self, X_train, y_train, shuffle=True):
        self._X_train = X_train
        self._y_train = y_train

        # Get the numbers of examples and inputs.
        self._n_examples = self._X_train.shape[0]
        self._n_inputs = self._X_train.shape[1]

        idx = list(range(self._n_examples))
        if shuffle:
            random.shuffle(idx)
        self._X_train = self._X_train[idx]
        self._y_train = self._y_train[idx]
    
    def _create_placeholders(self):
        self._X = tf.placeholder(tf.float32, shape=(self._batch_size, self._n_inputs), name='X')
        self._y = tf.placeholder(tf.float32, shape=(self._batch_size, 1), name='y')
    
    def _create_weights(self):
        self._w = tf.get_variable(shape=(self._n_inputs, 1), 
                                  initializer=tf.random_normal_initializer(0, 0.01), 
                                  name='weights')
        self._b = tf.get_variable(shape=(1, 1), 
                                  initializer=tf.zeros_initializer(), name='bias')
    
    def _create_model(self):
        self._y_pred = tf.add(tf.matmul(self._X, self._w), self._b, name='y_pred')
    
    def _create_loss(self):
        # Mean squared error as loss.
        self._error = self._y_pred - self._y
        self._loss = tf.reduce_mean(tf.square(self._error), name='loss')
    
    def _create_optimizer(self):
        # Applt gradient descent optimization.
        self._optimizer = (
            tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            .minimize(self._loss))

    def build_graph(self):
        self._create_placeholders()
        self._create_weights()
        self._create_model()
        self._create_loss()
        self._create_optimizer()

    def _fetch_batch(self):
        idx = list(range(self._n_examples))
        for i in range(0, self._n_examples, self._batch_size):
            idx_batch = idx[i:min(i + self._batch_size, self._n_examples)]
            yield (self._X_train[idx_batch, :], self._y_train[idx_batch, :])

    def train_model(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self._n_epochs):
                total_loss = 0

                for X_train_b, y_train_b in self._fetch_batch():
                    _, batch_loss = sess.run([self._optimizer, self._loss],
                                             feed_dict={self._X: X_train_b, self._y: y_train_b})
                    total_loss += batch_loss

                if epoch % 100 == 0:
                    print('Epoch {0}: training loss: {1}'
                          .format(epoch, total_loss / self._n_examples))

            w_out, b_out = sess.run([self._w, self._b])
            print('Weight: {}'.format(w_out))
            print('Bias: {}'.format(b_out))


def main():
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    # Read California housing data.
    housing = fetch_california_housing()

    data = housing.data
    label = housing.target.reshape(-1, 1)

    # Important: Normalize features first.
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split data into training/test datasets.
    test_ratio = 0.2
    test_size = int(data.shape[0] * test_ratio)

    X_train = data[:-test_size]
    X_test = data[-test_size:]
    y_train = label[:-test_size]
    y_test = label[-test_size:]

    # Train Linear Regression model.
    reset_graph()

    linreg = LinearRegression()
    linreg.get_dataset(X_train, y_train)
    linreg.build_graph()
    linreg.train_model()


if __name__ == '__main__':
    main()
