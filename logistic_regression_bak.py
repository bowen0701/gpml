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


def initialize_weights(dim):
    """Initialize weights.

    This function creates a vector of zeros of shape (dim, 1) for w and b to 0.
    
    Args:
      dim: A integer. Size of the w vector (or number of parameters.)
    
    Returns:
      w: A Numpy array. Initialized vector of shape (dim, 1)
      b: A integer. Initialized scalar (corresponds to the bias)
    """
    w = np.zeros(dim).reshape(dim, 1)
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def activation(w, b, X):
    """Activation function using sigmoid function."""
    A = sigmoid(np.dot(X, w) + b)
    return A

def cross_entropy(y, A, m):
    """Cross entropy."""
    cross_entropy = - 1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    return cross_entropy

def gradient(X, y, A, m):
    """Gradient for weight vector and bias."""
    dw = 1 / m * np.dot(X.T, (A - y))
    db = 1 / m * np.sum(A - y)
    return dw, db

def propagate(w, b, X, y):
    """Forward & backward propagation.

    Implement the cost function and its gradient for the propagation.

    Args:
      w: A Numpy array. Weights of size (num_px * num_px * 3, 1)
      b: A float. Bias.
      X: A Numpy array. Data of size (number of examples, num_px * num_px * 3).
      y: A Numpy array. True "label" vector (containing 0 or 1) 
         of size (number of examplesm, 1).

    Returns:
      cost: A float. Negative log-likelihood cost for logistic regression.
      dw: A Numpy array. Gradient of the loss w.r.t. w, thus same shape as w.
      db: A float. Gradient of the loss w.r.t b, thus same shape as b.
    """
    m = X.shape[0]
    y = y.reshape((m, 1))

    # Forward propagation from X to cost.
    # Compute activation.
    A = activation(w, b, X)
    # Compute cost.
    cost = cross_entropy(y, A, m)
    
    # Backward propagation to find gradient.
    dw, db = gradient(X, y, A, m)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db} 

    return grads, cost


def gradient_descent(w, b, X, y, num_iterations, learning_rate, print_cost=True):
    """Optimize using gradient descent.

    This function optimizes w and b by running a gradient descent algorithm.
    That is, write down two steps and iterate through them:
      1. Calculate the cost and the gradient for the current parameters. 
        Use propagate().
      2. Update the parameters using gradient descent rule for w and b.
    
    Args:
      w: A Numpy array. Weights of size (num_px * num_px * 3, 1).
      b: A scalar. Bias.
      X: A Numpy array. Data of shape (number of examples, num_px * num_px * 3).
      y: A Numpy array. True "label" vector (containing 0 if non-cat, 1 if cat), 
        of shape (number of examples, 1)
      num_iterations: A integer. Number of iterations of the optimization loop.
      learning_rate: A scalr. Learning rate of the gradient descent update rule.
      print_cost: A Boolean. Print the loss every 100 steps. Default: True.
    
    Returns:
      params: A dictionary containing the weights w and bias b.
      grads: A dictionary containing the gradients of the weights and bias 
        with respect to the cost function
      costs: A list of all the costs computed during the optimization, 
        this will be used to plot the learning curve.
    """   
    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, y)
        
        # Retrieve derivatives from grads
        dw = grads.get('dw')
        db = grads.get('db')
        
        # Update rule.
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Record the costs every 200 training examples and print.
        if i % 200 == 0:
            costs.append(cost)
        if print_cost and i % 200 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    """Prediction.

    Predict whether the label is 0 or 1 using learned logistic regression 
    parameters (w, b)
    
    Args:
      w: A Numpy array. Learned weights of size (num_px * num_px * 3, 1).
      b: A scalar. Learned bias.
      X: A Numpy array. New data of size (num_px * num_px * 3, number of examples).
    
    Returns:
      y_pred: A Numpy array containing all predictions (0/1) 
        for the examples in X.
    """
    m = X.shape[0]
    y_pred = np.zeros((m, 1))
    
    # Compute vector "A" predicting the probabilities of a label 1.
    A = activation(w, b, X)
    
    for i in range(A.shape[0]):
        # Convert probabilities a[i] to actual predictions y_pred[i].
        if A[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    
    assert(y_pred.shape == (m, 1))
    
    return y_pred


def accuracy(y_pred, y):
    acc = 1 - np.mean(np.abs(y_pred - y))
    return acc


def logistic_regression(X_train, y_train, X_test, y_test, 
                        num_iterations=2000, learning_rate=0.5, print_cost=True):
    '''Wrap-up function for logistic regression.

    Builds the logistic regression model by calling the function 
    you've implemented previously.
    
    Args:
      X_train: A Numpy. Training set of shape (m_train, num_px * num_px * 3).
      y_train: A Numpy array. Training labels of shape (m_train, 1).
      X_test: A Numpy array. Test set of shape (m_test, num_px * num_px * 3).
      y_test: A Numpy array. Test labels of shape (m_test, 1).
      num_iterations: An integer. Hyperparameter for the number of iterations 
        to optimize the parameters. Default: 2000.
      learning_rate: A scalar. Hyperparameter for the learning rate used 
        in the update rule of optimize(). Default: 0.005.
      print_cost: A Boolean. Print the cost every 100 iterations. Default: True.
    
    Returns:
      d: A dictionary containing information about the model.
    '''    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_weights(X_train.shape[1])

    # Gradient descent.
    parameters, grads, costs = gradient_descent(
        w, b, X_train, y_train, 
        num_iterations=num_iterations, learning_rate=learning_rate, 
        print_cost=print_cost)
    
    # Retrieve parameters w and b from dictionary 'parameters'
    w = parameters.get('w')
    b = parameters.get('b')
    
    # Predict test/train set examples (≈ 2 lines of code)
    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)

    # Print train/test Errors
    print('Train accuracy: {} %'
          .format(accuracy(y_pred_train.ravel(), y_train) * 100))
    print('Test accuracy: {} %'
          .format(accuracy(y_pred_test.ravel(), y_test) * 100))
    
    d = {'costs': costs,
         'y_pred_train': y_pred_train, 
         'y_pred_test': y_pred_test, 
         'w': w, 
         'b': b,
         'learning_rate' : learning_rate,
         'num_iterations': num_iterations}
    return d
