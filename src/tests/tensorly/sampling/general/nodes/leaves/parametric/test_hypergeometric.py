import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import Hypergeometric
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestHypergeometric(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(tl_isclose(samples.mean(axis=0), tl.tensor(n * M) / N, atol=0.01, rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        samples = sample(hypergeometric, 100000)

        self.assertTrue(tl_isclose(samples.mean(axis=0), tl.tensor(n * M) / N, atol=0.01, rtol=0.1))

    def test_sampling_5(self):

        hypergeometric = Hypergeometric(Scope([0]), 2, 2, 2)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            hypergeometric,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
