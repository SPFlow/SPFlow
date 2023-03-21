import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import Binomial
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestBinomial(unittest.TestCase):
    def test_sampling_1(self):

        # ----- p = 0 -----

        binomial = Binomial(Scope([0]), 10, 0.0)

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(binomial, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~tl_isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # ----- p = 1 -----

        binomial = Binomial(Scope([0]), 10, 1.0)

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(binomial, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~tl_isnan(samples)] == 10))

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- p = 0.5 -----

        binomial = Binomial(Scope([0]), 10, 0.5)

        samples = sample(binomial, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(5.0), rtol=0.1))

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
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
