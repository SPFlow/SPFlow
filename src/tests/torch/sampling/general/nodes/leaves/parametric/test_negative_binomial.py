import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import NegativeBinomial


class TestNegativeBinomial(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sampling_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- n = 1, p = 1.0 -----

        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(
            negative_binomial, data, sampling_ctx=SamplingContext([0, 2])
        )

        self.assertTrue(
            all(samples.isnan() == torch.tensor([[False], [True], [False]]))
        )
        self.assertTrue(all(samples[~samples.isnan()] == 0.0))

    def test_sampling_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- n = 10, p = 0.3 -----

        negative_binomial = NegativeBinomial(Scope([0]), 10, 0.3)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(
            torch.isclose(
                samples.mean(), torch.tensor(10 * (1 - 0.3) / 0.3), rtol=0.1
            )
        )

    def test_sampling_3(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- n = 5, p = 0.8 -----

        negative_binomial = NegativeBinomial(Scope([0]), 5, 0.8)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(
            torch.isclose(
                samples.mean(), torch.tensor(5 * (1 - 0.8) / 0.8), rtol=0.1
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
