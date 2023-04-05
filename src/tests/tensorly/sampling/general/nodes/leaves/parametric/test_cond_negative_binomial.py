import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import CondNegativeBinomial
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondNegativeBinomial(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 1, p = 1.0 -----

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda data: {"p": 1.0})
        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(negative_binomial, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~tl_isnan(samples)] == 0.0))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 10, p = 0.3 -----

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=10, cond_f=lambda data: {"p": 0.3})

        samples = sample(negative_binomial, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(10 * (1 - 0.3) / 0.3), rtol=0.1))

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- n = 5, p = 0.8 -----

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=5, cond_f=lambda data: {"p": 0.8})

        samples = sample(negative_binomial, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(5 * (1 - 0.8) / 0.8), rtol=0.1))

    def test_sampling_4(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=5, cond_f=lambda data: {"p": 0.8})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            negative_binomial,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
