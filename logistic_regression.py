from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

np.random.seed(71)


class LogisticRegression(object):
    """Numpy implementation of Logistic Regression."""

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

    def _logit(self, X):
        """Logit: unnormalized log probability."""
        return np.matmul(X, self.w) + self.b

    def _sigmoid(self, logit):
        """Sigmoid function by stabilization trick.

        sigmoid(z) = 1 / (1 + exp(-z)) 
                   = exp(z) / (1 + exp(z)) * exp(z_max) / exp(z_max)
                   = exp(z - z_max) / (exp(-z_max) + exp(z - z_max)),
        where z is the logit, and z_max = z - max(0, z).
        """
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        return np.exp(logit_stable) / (np.exp(-logit_max) + np.exp(logit_stable))
    
    def _model(self, X):
        """Logistic regression model."""
        logit = self._logit(X)
        return self._sigmoid(logit)

    def _loss(self, y, logit):
        """Cross entropy loss by stabilizaiton trick.

        cross_entropy_loss(y, z) 
          = - 1/n * \sum_{i=1}^n y_i * log p(y_i = 1|x_i) + (1 - y_i) * log p(y_i = 0|x_i)
          = - 1/n * \sum_{i=1}^n y_i * (z_i - log(1 + exp(z_i))) + (1 - y_i) * (-log(1 + exp(z_i))),
        where z is the logit, z_max = z - max(0, z),
          log p(y = 1|x)
            = log (1 / (1 + exp(-z))) 
            = log (exp(z) / (1 + exp(z)))
            = z - log(1 + exp(z))
        and 
          log(1 + exp(z)) := logsumexp(z)
            = log(exp(0) + exp(z))
            = log(exp(0) + exp(z) * exp(z_max) / exp(z_max))
            = z_max + log(exp(-z_max) + exp(z - z_max)).
        """
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        logsumexp_stable = logit_max + np.log(np.exp(-logit_max) + np.exp(logit_stable))
        self.cross_entropy = -(y * (logit - logsumexp_stable) + (1 - y) * (-logsumexp_stable))
        return np.mean(self.cross_entropy)

    def _optimize(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_ = self._model(X) 
        dw = 1 / m * np.matmul(X.T, y_ - y)
        db = np.mean(y_ - y)

        for (param, grad) in zip([self.w, self.b], [dw, db]):
            param[:] = param - self.lr * grad

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
                train_loss = self._loss(y_train_b, self._logit(X_train_b))
                total_loss += train_loss * X_train_b.shape[0]

            if epoch % 100 == 0:
                print('epoch {0}: training loss {1}'
                      .format(epoch, total_loss / self.n_examples))

        return self

    def get_coeff(self):
        return self.b, self.w.reshape((-1,))

    def predict(self, X):
        return self._model(X).reshape((-1,))


# TODO: Implement linear regression in PyTorch.


def reset_tf_graph(seed=71):
    """Reset default TensorFlow graph."""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class LogisticRegressionTF(object):
    """A TensorFlow implementation of Logistic Regression."""

    def __init__(self, batch_size=64, learning_rate=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def get_data(self, X_train, y_train, shuffle=True):
        """Get dataset and information."""
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
        """Create placeholder for features and labels."""
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

    def _logit(self, X):
        """Logit: unnormalized log probability."""
        return tf.matmul(X, self.w) + self.b

    def _model(self, X):
        """Logistic regression model."""
        logits = self._logit(X)
        return tf.math.sigmoid(logits)

    def _create_model(self):
        # Create logistic regression model.
        self.logits = self._logit(self.X)

    def _create_loss(self):
        # Create cross entropy loss.
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y,
            logits=self.logits,
            name='cross_entropy')
        self.loss = tf.reduce_mean(self.cross_entropy, name='loss')

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
            yield (self.X_train[idx_batch, :], 
                   self.y_train[idx_batch].reshape(-1, 1))

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
            saver.save(sess, 'checkpoints/logreg')

    def get_coeff(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load model.
            saver = tf.train.Saver()
            saver.restore(sess, 'checkpoints/logreg')
            return self.b.eval(), self.w.eval().reshape((-1,))

    def predict(self, X):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load model.
            saver = tf.train.Saver()
            saver.restore(sess, 'checkpoints/logreg')
            return self._model(X).eval().reshape((-1,))


# TODO: Implement logistic regression in MXNet.

# TODO: Implement logistic regression in Gluon.


def main():
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn
    from metrics import accuracy

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

    # Feature engineering for standardizing features by min-max scaler.
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_raw)
    X_test = min_max_scaler.transform(X_test_raw)

    # Train Numpy logistic regression model.
    print("Fit logreg in NumPy.")
    logreg = LogisticRegression(batch_size=64, lr=1, n_epochs=1000)
    logreg.get_data(X_train, y_train, shuffle=True)
    logreg.fit()

    p_pred_train = logreg.predict(X_train)
    y_pred_train = (p_pred_train > 0.5) * 1
    print('Training accuracy: {}'.format(accuracy(y_train, y_pred_train)))
    p_pred_test = logreg.predict(X_test)
    y_pred_test = (p_pred_test > 0.5) * 1
    print('Test accuracy: {}'.format(accuracy(y_test, y_pred_test)))

    # Train TensorFlow linear regression model.
    print("Fit logreg in TensorFlow.")
    reset_tf_graph()
    logreg_tf = LogisticRegressionTF(
        batch_size=64, learning_rate=1, n_epochs=1000)
    logreg_tf.get_data(X_train, y_train, shuffle=True)
    logreg_tf.build_graph()
    logreg_tf.fit()

    p_train_ = logreg_tf.predict(X_train)
    y_train_ = (p_train_ > 0.5) * 1
    print('Training accuracy: {}'.format(accuracy(y_train, y_pred_train)))
    p_test_ = logreg_tf.predict(X_test)
    y_test_ = (p_test_ > 0.5) * 1
    print('Test accuracy: {}'.format(accuracy(y_test, y_pred_test)))

    # Benchmark with sklearn's Logistic Regression.
    print('Benchmark with logreg in Scikit-Learn.')
    logreg_sk = LogisticRegressionSklearn(C=1e4, solver='lbfgs', max_iter=500)
    logreg_sk.fit(X_train, y_train)

    p_train_ = logreg_sk.predict(X_train)
    y_train_ = (p_train_ > 0.5) * 1
    print('Training accuracy: {}'.format(accuracy(y_train, y_train_)))
    p_test_ = logreg_sk.predict(X_test)
    y_test_ = (p_test_ > 0.5) * 1
    print('Test accuracy: {}'.format(accuracy(y_test, y_test_)))


if __name__ == '__main__':
    main()
