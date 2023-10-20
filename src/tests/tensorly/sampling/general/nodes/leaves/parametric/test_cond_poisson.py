import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import CondPoisson
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondPoisson(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- l = 1.0 -----

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(poisson, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

        samples = sample(poisson, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(1.0), rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- l = 0.5 -----

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 0.5})

        samples = sample(poisson, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(0.5), rtol=0.1))

    def test_sampling_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- l = 2.5 -----

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 2.5})

        samples = sample(poisson, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor(2.5), rtol=0.1))

    def test_sampling_4(self):

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 2.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            poisson,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()