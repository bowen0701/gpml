from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import tensorflow as tf

np.random.seed(71)


class LogisticRegression(object):
    """Numpy implementation of Logistic Regression."""
    def __init__(self, batch_size=64, lr=0.01, n_epochs=1000):
        self._batch_size = batch_size
        self._lr = lr
        self._n_epochs = n_epochs

    def get_dataset(self, X_train, y_train, shuffle=True):
        """Get dataset information."""
        self._X_train = X_train
        self._y_train = y_train

        # Get the numbers of examples and inputs.
        self._n_examples = self._X_train.shape[0]
        self._n_inputs = self._X_train.shape[1]

        if shuffle:
            idx = list(range(self._n_examples))
            random.shuffle(idx)
            self._X_train = self._X_train[idx]
            self._y_train = self._y_train[idx]

    def _create_weights(self):
        """Create model weights and bias."""
        self._w = np.zeros(self._n_inputs).reshape(self._n_inputs, 1)
        self._b = np.zeros(1).reshape(1, 1)

    def _sigmoid(self, logit):
        """Sigmoid function (stable version)."""
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        return np.exp(logit_stable) / (np.exp(-logit_max) + np.exp(logit_stable))

    def _logit(self, X):
        return np.matmul(X, self._w) + self._b
    
    def _model(self, X):
        """Logistic regression model (stable version)."""
        logit = self._logit(X)
        return self._sigmoid(logit)

    def _loss(self, y, logit):
        """Cross entropy loss (stable version) by subtracting the maximum of (0, logit)."""
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        logsumexp_stable = logit_max + np.log(np.exp(-logit_max) + np.exp(logit_stable))
        self._cross_entropy = -(y * (logit - logsumexp_stable) + (1 - y) * (-logsumexp_stable))
        return np.mean(self._cross_entropy)

    def _optimizer(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_hat = self._model(X) 
        dw = -1 / m * np.matmul(X.T, y - y_hat)
        db = -np.mean(y - y_hat)
        
        for (param, grad) in zip([self._w, self._b], [dw, db]):
            param[:] = param - self._lr * grad

    def build_graph(self):
        self._create_weights()
            
    def _fetch_batch(self):
        idx = list(range(self._n_examples))
        for i in range(0, self._n_examples, self._batch_size):
            idx_batch = idx[i:min(i + self._batch_size, self._n_examples)]
            yield (self._X_train.take(idx_batch, axis=0), self._y_train.take(idx_batch, axis=0))

    def fit(self):
        for epoch in range(self._n_epochs):
            total_loss = 0
            for X_train_b, y_train_b in self._fetch_batch():
                y_train_b = y_train_b.reshape((y_train_b.shape[0], -1))
                self._optimizer(X_train_b, y_train_b)
                train_loss = self._loss(y_train_b, self._logit(X_train_b))
                total_loss += train_loss * X_train_b.shape[0]

            if epoch % 100 == 0:
                print('epoch {0}: training loss {1}'.format(epoch, total_loss))

        return self

    def get_coeff(self):
        return self._b, self._w.reshape((-1,))

    def predict(self, X_test):
        return self._model(X_test).reshape((-1,))


# Reset default graph.
def reset_tf_graph(seed=71):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class LogisticRegressionTF(object):
    """A TensorFlow implementation of Logistic Regression."""
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
        self._X = tf.placeholder(tf.float32, shape=(None, self._n_inputs), name='X')
        self._y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    
    def _create_weights(self):
        self._w = tf.get_variable(shape=(self._n_inputs, 1), 
                                  initializer=tf.random_normal_initializer(0, 0.01), 
                                  name='weights')
        self._b = tf.get_variable(shape=(1, 1), 
                                  initializer=tf.zeros_initializer(), name='bias')
    
    def _create_model(self):
        # Logistic regression model.
        self._logit = tf.add(tf.matmul(self._X, self._w), self._b, name='logit')
        self._logreg = tf.math.sigmoid(self._logit, name='logreg')

    def _create_loss(self):
        # Cross entropy loss.
        self._cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self._y,
            logits=self._logit,
            name='y_pred')   
        self._loss = tf.reduce_mean(self._cross_entropy, name='loss')

    def _create_optimizer(self):
        # Gradient descent optimization.
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

    def fit(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self._n_epochs):
                total_loss = 0
                for X_train_b, y_train_b in self._fetch_batch():
                    feed_dict = {self._X: X_train_b, self._y: y_train_b}
                    _, batch_loss = sess.run([self._optimizer, self._loss],
                                             feed_dict=feed_dict)
                    total_loss += batch_loss * X_train_b.shape[0]

                if epoch % 100 == 0:
                    print('Epoch {0}: training loss: {1}'
                          .format(epoch, total_loss / self._n_examples))

            w_out, b_out = sess.run([self._w, self._b])
            print('Weight:\n{}'.format(w_out))
            print('Bias: {}'.format(b_out))


def main():
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    breast_cancer = load_breast_cancer()
    data = breast_cancer.data
    label = breast_cancer.target.reshape(-1, 1)

    # Normalize features first.
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split data into training/test data.
    test_ratio = 0.2
    test_size = int(data.shape[0] * test_ratio)

    X_train = data[:-test_size]
    X_test = data[-test_size:]
    y_train = label[:-test_size]
    y_test = label[-test_size:]

    # TODO: Train Numpy linear regression model.

    # Train TensorFlow logistic regression model.
    reset_tf_graph()

    logreg = LogisticRegressionTF()
    logreg.get_dataset(X_train, y_train)
    logreg.build_graph()
    logreg.fit()


if __name__ == '__main__':
    main()
