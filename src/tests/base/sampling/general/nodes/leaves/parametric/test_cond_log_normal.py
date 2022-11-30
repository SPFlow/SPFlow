import random
import unittest

import numpy as np

from spflow.base.sampling import sample
from spflow.base.structure.spn import CondLogNormal
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestCondLogNormal(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 0.0, std = 1.0 -----

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 1.0})

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))

        samples = sample(log_normal, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.exp(0.0 + (1.0**2 / 2.0)), rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- mean = 1.0, std = 0.5 -----

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 1.0, "std": 0.5})

        samples = sample(log_normal, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.exp(1.0 + (0.5**2 / 2.0)), rtol=0.1))

    def test_sampling_5(self):

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 1.0, "std": 0.5})

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            log_normal,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
