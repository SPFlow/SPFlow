import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import Gaussian


class TestGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        gaussian = Gaussian(Scope([0]), 0.0, 0.0005)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(gaussian, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        # ----- verify samples -----
        samples = sample(gaussian, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.Tensor([0.0]), atol=0.01))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
