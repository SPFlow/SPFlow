import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import CondGamma
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondGamma(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- alpha = 1, beta = 1 -----

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(gamma, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

        samples = sample(gamma, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(1.0 / 1.0), rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- alpha = 0.5, beta = 1.5 -----

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 0.5, "beta": 1.5})

        samples = sample(gamma, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(0.5 / 1.5), rtol=0.1))

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- alpha = 1.5, beta = 0.5 -----

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.5, "beta": 0.5})

        samples = sample(gamma, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(1.5 / 0.5), rtol=0.1))

    def test_sampling_4(self):

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.5, "beta": 0.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            gamma,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
