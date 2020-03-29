from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Tensor(object):
    """Automatic differentiation."""

    def __init__(self, x, requires_grad=False, dependency=None):
        self._x = x
        self._shape = self._x.shape
        self._grad = None

        if requires_grad:
            self.zero_grad()
        self._require_grad = requires_grad

        if not dependency:
            dependency = []
        self._dependency = dependency

    @property
    def values(self):
        return self._x

    @values.setter
    def values(self, x_new):
        self._x = np.array(x_new)
        self._grad = None

    def zero_grad(self):
        self.grad = np.zeros(self._shape)

    # TODO: Continue implementing autograd.


def main():
    pass


if __name__ == '__main__':
    main()
