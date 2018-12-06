from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sigmoid(x):
    """Compute the sigmoid of x.

    Args:
      x: A scalar or numpy array of any size.

    Returns:
      s: sigmoid(x).
    """
    s = 1 / (1 + np.exp(-x))    
    return s


class LogisticRegression(object):
    """Logistic regression class."""
    def __init__(self, learning_rate=0.5, num_iter=2000, print_cost=True):
        """Create a `LogisticRegression` class.

        Args:

        Returns:

        """
        self._learning_rate = learning_rate
        self._num_iter = num_iter
        self._print_cost = print_cost

    def _initialize_coeffs(self):
        """Initialize coefficients, including weights and bias.

        This function creates 
          - a zero weights w of shape (dim, 1).
          - a 0 for bias b.
        
        Returns:
          w: A Numpy array. Initialized weights.
          b: A integer. Initialized bias.
        """
        dim = self._X_train.shape[1]
        w = np.zeros(dim).reshape(dim, 1)
        b = 0
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def _propagate(self, w, b):
        """Forward & backward propagation.

        Implement the cost function and its gradient for the propagation.

        Args:
          w: A Numpy array. Weights.
          b: A float. Bias.

        Returns:
          cost: A float. Negative log-likelihood cost for logistic regression.
          dw: A Numpy array. Gradient of the loss w.r.t. w, thus same shape as w.
          db: A float. Gradient of the loss w.r.t b, thus same shape as b.
        """
        m = self._X_train.shape[1]
        
        # Forward propagation from X to cost.
        # Compute activation.
        A = sigmoid(np.dot(w.T, self._X_train) + b)
        
        # Backward propagation to find gradient.
        dw = 1 / m * np.dot(self._X_train, (A - self._y_train).T)
        db = 1 / m * np.sum(A - self._y_train)
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        grads = {'dw': dw,
                 'db': db}

        # Compute cost.
        cost = - 1 / m * np.sum(
            self._y_train * np.log(A) + (1 - self._y_train) * np.log(1 - A))
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return grads, cost

    def _gradient_descent(self, w, b):
        """Optimization function.

        This function optimizes (w, b) by running a gradient descent algorithm.
        That is, write down two steps and iterate through them:
          - Calculate the cost and the gradient for the current parameters. 
            Use propagate().
          - Update the parameters using gradient descent rule for (w, b).

        Args:
          w: A Numpy array. Initialized weights.
          b: A integer. Initialized bias.

        Returns:
          params: A dictionary containing the weights and bias (w, b).
          grads: A dictionary containing the gradients of the (w, b) 
            with respect to the cost function.
          costs: A list of all the costs computed during the optimization, 
            this will be used to plot the learning curve.
        """   
        costs = []

        for i in range(self._num_iterations):
            # Cost and gradient calculation (≈ 1-4 lines of code)
            grads, cost = _propagate(self, w, b)
            
            # Retrieve derivatives from grads
            dw = grads.get('dw')
            db = grads.get('db')
            
            # Update rule.
            w -= self._learning_rate * dw
            b -= self._learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            # Print the cost every 100 training examples
            if self._print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
        
        coeffs = {'w': w,
                  'b': b}
        
        grads = {'dw': dw,
                 'db': db}
        
        return coeffs, grads, costs

    def fit(self, X_train, y_train):
        """Fit logist regression.

        Args:

        Returns:

        """
        self._X_train = X_train
        self._y_train = y_train

        # Initialize parameters with zeros.
        w, b = _initialize_coeffs(self)

        # Optimize using gradient descent.
        coeffs, grads, costs = _gradient_descent(self, w, b)

        pass

    # TODO: Continue implementing LogisticRegression class.