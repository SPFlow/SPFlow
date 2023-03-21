import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import CondLogNormal
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondLogNormal(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 0.0, std = 1.0 -----

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 1.0})

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

        samples = sample(log_normal, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.exp(0.0 + (1.0**2 / 2.0)), rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 1.0, std = 0.5 -----

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 1.0, "std": 0.5})

        samples = sample(log_normal, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.exp(1.0 + (0.5**2 / 2.0)), rtol=0.1))

    def test_sampling_5(self):

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 1.0, "std": 0.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            log_normal,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
