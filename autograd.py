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
        self._requires_grad = requires_grad

        if not dependency:
            dependency = []
        self._dependency = dependency

    @property
    def values(self):
        """Values getter."""
        return self._x

    @values.setter
    def values(self, x_new):
        """Values setter."""
        self._x = np.array(x_new)
        self._grad = None

    @staticmenthod
    def as_tensor(obj):
        """Convert object to Tensor."""
        if not isinstance(obj, Tensor):
            obj = Tensor(obj)
        return obj

    @staticmethod
    def _build_binary_ops_tensor(ts1, ts2, x, grad_fn_ts1, grad_fn_ts2):
        """Build binary operator tensor."""
        requires_grad = ts1._requires_grad or ts2._requires_grad
        dependency = []

        if ts1._requires_grad:
            dependency.append(dict(tensor=ts1, grad_fn=grad_fn_ts1))
        if ts2._requires_grad:
            dependency.append(dict(tensor=ts2, grad_fn=grad_fn_ts2))
        return Tensor(x, requires_grad=requires_grad, dependency=dependency)

    @staticmethod
    def _matmul(ts1, ts2):
        """Util function for matrix multiplication.

        X = X1 @ X2
        dX/dX1 = grad @ X2^T
        dx/dX2 = X1^T @ grad
        """
        x = ts1._x @ ts2._x

        def grad_fn_ts1(grad):
            return grad @ ts2._x.T
        def grad_fn_ts2(grad):
            return ts1._x.T @ grad
        return _build_binary_ops_tensor(ts1, ts2, x, grad_fn_ts1, grad_fn_ts2)

    def zero_grad(self):
        """Initialize gradients with zeros."""
        self.grad = np.zeros(self._shape)

    def __matmul__(self, other):
        """Matrix multiplication: self & other."""
        return _matmul(self, as_tensor(other))

    # TODO: Continue implementing autograd.


def main():
    pass


if __name__ == '__main__':
    main()
