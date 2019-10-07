"""Sample random variable."""

from __future__ import absolute_import, division, print_function

import random


class SampleDiscrete(object):
    def __init__(self, values, probs=None, n_bins=1000):
        """Preprocess inputs for sampling from "uniform" duplicated values.

        Apply the Probability Integral Transform for uniform discrete r.v.,
        [x_1, ..., x_n] with probs 1/n:
        X = int(nU) + 1, where U ~ Uniform(0, 1).

        From the above we can use the following approach to sample discrete r.v.:
        - For r.v. with equal probs, it follows trivially.
        - For r.v. with unequal probs, preprocess to "uniform" r.v. with 1/n.
          Specifically, preprocess [x_1, x_2, ...] with [p_1, p_2, ...] to
          duplicated values [x_1, x_1, ..., x_2, x_2, ...] with frequency based on probs.

        Args:
          values: A list. Values from which we want to sample.
          probs: A list. Sampling probabilities for values. Default: None.
          n_bins: A int. Bin number to duplicate values for uniform r.v. Default: 1000.
        """
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
        """Sample from values with probs."""
        # Sample a r.v. from Uniform(0, 1).
        u = random.uniform(0, 1)

        # Get index = int(n*u) since index starts from 0.
        i = int(self.n * u)
        return self.values[i]


def main():
    import numpy as np

    n_sim = 10000

    # Sample discrete random variable with equal probs.
    # Output: should be close to 0.5
    values = [0, 1]

    sample_discrete = SampleDiscrete(values)
    sampled_rvs = [None] * n_sim
    for i in range(n_sim):
        sampled_rvs[i] = sample_discrete.sample()
    print(np.mean(sampled_rvs))

    # Sample discrete random variable with unequal probs.
    # Output: should be close to 0.7
    values = [0, 1, 2]
    probs = [0.5, 0.3, 0.2]

    sample_discrete = SampleDiscrete(values, probs)
    sampled_rvs = [None] * n_sim
    for i in range(n_sim):
        sampled_rvs[i] = sample_discrete.sample()
    print(np.mean(sampled_rvs))


if __name__ == '__main__':
    main()
