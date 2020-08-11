from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def inner_product(x, y):
    """Inner product."""
    return x.dot(y)


def inner_product_d(x_d, y_d):
    """Inner product for sparse vectors."""
    inner_prod = 0.0
    for k, v in x_d.items():
        if k in y_d:
            inner_prod += v * y_d[k]
    return inner_prod


def correlation_coeff(x, y):
    """Correlation coefficient.

    Note: To speed up computation, we drop normalized term (n - 1)**(-1),
    since this term is redundant after division with sqrt.
    """
    x_ = x - x.mean()
    y_ = y - y.mean()
    cov = inner_product(x_, y_)
    var_x = inner_product(x_, x_)
    var_y = inner_product(y_, y_)
    return cov / math.sqrt(var_x * var_y)


def cosine_similarity(x, y):
    """Cosine similarity."""
    inner_prod = inner_product(x, y)
    norm2_x = inner_product(x, x)
    norm2_y = inner_product(y, y)
    return inner_prod / math.sqrt(norm2_x * norm2_y)


def cosine_similarity_d(x_d, y_d):
    """Cosine similarity for sparse vectors."""
    inner_prod = inner_product_d(x_d, y_d)
    x = np.array(list(x_d.values()))
    y = np.array(list(y_d.values()))
    norm2_x = inner_product(x, x)
    norm2_y = inner_product(y, y)
    return inner_prod / math.sqrt(norm2_x * norm2_y)


def accuracy(y_true, y_pred):
    """Accuracy."""
    return (y_true == y_pred).mean()

