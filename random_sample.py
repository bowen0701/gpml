"""Sample random variable."""

from __future__ import absolute_import, division, print_function

import random


class SampleDiscreteRV(object):
    def __init__(self, values, probs=None, n_bins=1000):
        # Preprocess inputs for sampling from "uniform" duplicated values with 1/n.
        self.n_bins = n_bins

        if not probs:
            # For values with "equal" probs: 
            n_vals = len(values)
            probs = [1 / n_vals] * n_vals

        # Duplicate values by the corresponding binned probs.
        # E.g. for balanced coin: [0, 1] with [50, 50].
        binned_probs = [int(round(self.n_bins * p)) for p in probs]
        binned_values_ls = [[v] * p for v, p in zip(values, binned_probs)]
        binned_values = [v for ls in binned_values_ls for v in ls]

        self.n = len(binned_values)
        self.values = binned_values

    def sample(self):
        # Sample a r.v. from Uniform(0, 1).
        u = random.uniform(0, 1)

        # Get index = int(n*u), but not "int(n*u) + 1", since index starts from 0.
        i = int(self.n * u)
        return self.values[i]


def main():
    import numpy as np

    n_sim = 10000

    # Sample discrete random variable with equal probs.
    # Output: should be close to 0.5
    values = [0, 1]

    sample_discrete = SampleDiscreteRV(values)
    sampled_rvs = [None] * n_sim
    for i in range(n_sim):
        sampled_rvs[i] = sample_discrete.sample()
    print(np.mean(sampled_rvs))

    # Sample discrete random variable with unequal probs.
    # Output: should be close to 0.7
    values = [0, 1, 2]
    probs = [0.5, 0.3, 0.2]

    sample_discrete = SampleDiscreteRV(values, probs)
    sampled_rvs = [None] * n_sim
    for i in range(n_sim):
        sampled_rvs[i] = sample_discrete.sample()
    print(np.mean(sampled_rvs))


if __name__ == '__main__':
    main()
