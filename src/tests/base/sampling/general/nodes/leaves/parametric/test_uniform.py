from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.base.structure.spn import Uniform
from spflow.base.sampling import sample

import numpy as np
import unittest
import random


class TestUniform(unittest.TestCase):
    def test_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- a = -1.0, b = 2.5 -----

        uniform = Uniform(Scope([0]), -1.0, 2.5)
        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(uniform, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(
            all(np.isnan(samples) == np.array([[False], [True], [False]]))
        )

        samples = sample(uniform, 1000)
        self.assertTrue(all((samples >= -1.0) & (samples <= 2.5)))

    def test_sampling_2(self):

        uniform = Uniform(Scope([0]), -1.0, 2.5)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            uniform,
            np.array([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()