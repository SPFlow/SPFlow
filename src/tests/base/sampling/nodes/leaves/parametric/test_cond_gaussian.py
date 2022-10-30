from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.base.sampling.nodes.leaves.parametric.cond_gaussian import sample
from spflow.base.sampling.module import sample

import numpy as np
import random

import unittest


class TestCondGaussian(unittest.TestCase):
    def test_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": 0.0, "std": 0.0005}
        )

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(gaussian, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(
            all(np.isnan(samples) == np.array([[False], [True], [False]]))
        )

        # ----- verify samples -----
        samples = sample(gaussian, 1000)
        self.assertTrue(np.isclose(samples.mean(), np.array([0.0]), atol=0.01))

    def test_sampling_2(self):

        gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            gaussian,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
