from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class LogisticRegression(object):
    """Logistic regression class."""
    def __init__(self, batch_size=10, lr=0.01, n_epochs=5):
        """Numpy implemenation of Logistic Regression."""
        self._batch_size = batch_size
        self._lr = lr
        self._n_epochs = n_epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_grad(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _logreg(self, X, w, b):
        return self._sigmoid(np.dot(w.T, X) + b)

    def _cross_entropy(self, y_hat, y): 
        return -1 * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def _weights_init(self):
        w = np.zeros(self._n_inputs).reshape(self._n_inputs, 1)
        b = 0.0
        return w, b

    def _sgd(self, X, y, w, b):
        # TODO: Revise SGD method.
        m = X.shape[0]
        y_hat = self._logreg(X, w, b) 
        dw = 1 / m * np.dot(X, (y_hat - y).T)
        db = 1 / m * np.sum(A - self._y_train)
        
        for param in [w, b]:
            param[:] = param - self.lr * 

    # def _propagate(self, w, b):
    #     m = self._n_examples
        
    #     # Forward propagation from X to cost.
    #     # Compute activation.
    #     A = self._logreg(self._X_train, w, b)
        
    #     # Backward propagation to find gradient.
    #     dw = 1 / m * np.dot(self._X_train, (A - self._y_train).T)
    #     db = 1 / m * np.sum(A - self._y_train)
    #     grads = {'dw': dw, 'db': db}

    #     # Compute cost.
    #     cost = 
    #     cost = np.squeeze(cost)

    #     return grads, cost

    # def _gradient_descent(self, w, b): 
    #     costs = []

    #     for i in range(self._num_iterations):
    #         # Cost and gradient calculation (≈ 1-4 lines of code)
    #         grads, cost = _propagate(self, w, b)
            
    #         # Retrieve derivatives from grads
    #         dw = grads.get('dw')
    #         db = grads.get('db')
            
    #         # Update rule.
    #         w -= self._learning_rate * dw
    #         b -= self._learning_rate * db
            
    #         # Record the costs
    #         if i % 100 == 0:
    #             costs.append(cost)
    #         # Print the cost every 100 training examples
    #         if self._print_cost and i % 100 == 0:
    #             print("Cost after iteration %i: %f" %(i, cost))
        
    #     coeffs = {'w': w,
    #               'b': b}
        
    #     grads = {'dw': dw,
    #              'db': db}
        
    #     return coeffs, grads, costs

    def _data_iter(self):
        idx = list(range(self._n_examples))
        random.shuffle(idx)
        for i in range(0, self._n_examples, self._batch_size):
            idx_batch = np.array(idx[i:min(i + self._batch_size, self._n_examples)])
            yield self._X_train.take(idx_batch), self.y_train.take(idx_batch)

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        self._n_examples, self._n_inputs = X_train.shape

        logreg = self._logreg
        loss = self._cross_entropy
        w, b = self._weights_init()

        for epoch in range(n_epochs):
            for X, y in self._data_iter():
                y_hat = self._logreg(X, w, b)
                loss = self._cross_entropy(y_hat, y)

        pass

    def get_coeff(self):
        pass

    def predict(self, X_test):
        pass
