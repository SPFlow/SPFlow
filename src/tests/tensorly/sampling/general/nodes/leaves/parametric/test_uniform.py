import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves import Uniform
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestUniform(unittest.TestCase):
    def test_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # ----- a = -1.0, b = 2.5 -----

        uniform = Uniform(Scope([0]), -1.0, 2.5)
        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(uniform, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

        samples = sample(uniform, 1000)
        self.assertTrue(all((samples >= -1.0) & (samples <= 2.5)))

    def test_sampling_2(self):

        uniform = Uniform(Scope([0]), -1.0, 2.5)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            uniform,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
