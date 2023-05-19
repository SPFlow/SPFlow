import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_isclose

from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import GeneralGaussian as Gaussian
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


class TestGaussian(unittest.TestCase):
    def test_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        gaussian = Gaussian("numpy", Scope([0]), 0.0, 0.0005)

        data = tl.tensor([[tl.nan], [tl.nan], [tl.nan]])

        samples = sample(gaussian, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(tl_isnan(samples) == tl.tensor([[False], [True], [False]])))

        # ----- verify samples -----
        samples = sample(gaussian, 1000)
        self.assertTrue(tl_isclose(samples.mean(), tl.tensor([0.0]), atol=0.01))

    def test_sampling_2(self):

        gaussian = Gaussian("numpy", Scope([0]), 0.0, 1.0)

        # make sure that instance ids out of bounds raise errors
        self.assertRaises(
            ValueError,
            sample,
            gaussian,
            tl.tensor([[0]]),
            sampling_ctx=SamplingContext([1]),
        )


if __name__ == "__main__":
    unittest.main()
