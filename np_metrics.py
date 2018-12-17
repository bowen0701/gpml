from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def inner_product(x, y):
    return x.dot(y)


def cosine_similarity(x, y):
    numerator = inner_product(x, y)
    denominator = math.sqrt(inner_product(x, x) * inner_product(y, y))
    return numerator / denominator


def main():
    import time

    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])

    start_time = time.time()
    print('Inner product: {}'.format(inner_product(x, y)))
    print('Time: {}'.format(time.time() - start_time))

    start_time = time.time()
    print('Cosine similarity: {}'.format(cosine_similarity(x, y)))
    print('Time: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()
