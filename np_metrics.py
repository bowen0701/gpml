from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def inner_product(x, y):
    """Inner product."""
    return x.dot(y)


def correlation_coeff(x, y):
    """Correlation coefficient.

    Note: To speed up computation, we drop normalized term (n - 1)**(-1),
    since this term is redundant after division with sqrt.
    """
    x_cen = x - x.mean()
    y_cen = y - y.mean()
    cov = inner_product(x_cen, y_cen)
    var_x = inner_product(x_cen, x_cen)
    var_y = inner_product(y_cen, y_cen)
    return cov / math.sqrt(var_x * var_y)


def cosine_similarity(x, y):
    """Cosine similarity."""
    inner_prod = inner_product(x, y)
    norm2_x = inner_product(x, x)
    norm2_y = inner_product(y, y)
    return inner_prod / math.sqrt(norm2_x * norm2_y)


def main():
    import time

    x = np.array([1, 2, 3])
    y = np.array([2, 1.5, 0.5])

    start_time = time.time()
    print('Correlation coeff: {}'.format(correlation_coeff(x, y)))
    print('Time: {}'.format(time.time() - start_time))

    start_time = time.time()
    print('Cosine similarity: {}'.format(cosine_similarity(x, y)))
    print('Time: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()
