from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial,
)
from spflow.base.sampling.nodes.leaves.parametric.negative_binomial import (
    sample,
)
from spflow.base.sampling.module import sample

import numpy as np
import random

import unittest


class TestNegativeBinomial(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 1, p = 1.0 -----

        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(
            negative_binomial, data, sampling_ctx=SamplingContext([0, 2])
        )

        self.assertTrue(
            all(np.isnan(samples) == np.array([[False], [True], [False]]))
        )
        self.assertTrue(all(samples[~np.isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 10, p = 0.3 -----

        negative_binomial = NegativeBinomial(Scope([0]), 10, 0.3)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(
            np.isclose(samples.mean(), np.array(10 * (1 - 0.3) / 0.3), rtol=0.1)
        )

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 5, p = 0.8 -----

        negative_binomial = NegativeBinomial(Scope([0]), 5, 0.8)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(
            np.isclose(samples.mean(), np.array(5 * (1 - 0.8) / 0.8), rtol=0.1)
        )

    def test_sampling_4(self):

        negative_binomial = NegativeBinomial(Scope([0]), 5, 0.8)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            negative_binomial,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
