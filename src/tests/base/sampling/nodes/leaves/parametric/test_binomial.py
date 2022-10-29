from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.base.sampling.nodes.leaves.parametric.binomial import sample
from spflow.base.sampling.module import sample

import numpy as np
import random

import unittest


class TestBinomial(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 0 -----

        binomial = Binomial(Scope([0]), 10, 0.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(binomial, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(
            all(np.isnan(samples) == np.array([[False], [True], [False]]))
        )
        self.assertTrue(all(samples[~np.isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # ----- p = 1 -----

        binomial = Binomial(Scope([0]), 10, 1.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(binomial, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(
            all(np.isnan(samples) == np.array([[False], [True], [False]]))
        )
        self.assertTrue(all(samples[~np.isnan(samples)] == 10))

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- p = 0.5 -----

        binomial = Binomial(Scope([0]), 10, 0.5)

        samples = sample(binomial, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.array(5.0), rtol=0.1))

    def test_sampling_4(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        binomial = Binomial(Scope([0]), 5, 0.5)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            binomial,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
