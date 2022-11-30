import unittest

import numpy as np

from spflow.base.sampling import sample
from spflow.base.structure.spn import Bernoulli
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestBernoulli(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 0 -----

        bernoulli = Bernoulli(Scope([0]), 0.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # ----- p = 1 -----

        bernoulli = Bernoulli(Scope([0]), 1.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(bernoulli, data, sampling_ctx=SamplingContext([0, 2], [[0], [0]]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))
        self.assertTrue(all(samples[~np.isnan(samples)] == 1.0))

    def test_sampling_3(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            bernoulli,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
