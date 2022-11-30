from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.structure.spn import CondGamma
from spflow.torch.sampling import sample

import torch
import numpy as np
import random
import unittest


class TestGamma(unittest.TestCase):
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

        # ----- alpha = 1, beta = 1 -----

        gamma = CondGamma(
            Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0}
        )

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(gamma, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(
            all(samples.isnan() == torch.tensor([[False], [True], [False]]))
        )

        samples = sample(gamma, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(1.0 / 1.0), rtol=0.1)
        )

    def test_sampling_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- alpha = 0.5, beta = 1.5 -----

        gamma = CondGamma(
            Scope([0], [1]), cond_f=lambda data: {"alpha": 0.5, "beta": 1.5}
        )

        samples = sample(gamma, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(0.5 / 1.5), rtol=0.1)
        )

    def test_sampling_3(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # ----- alpha = 1.5, beta = 0.5 -----

        gamma = CondGamma(
            Scope([0], [1]), cond_f=lambda data: {"alpha": 1.5, "beta": 0.5}
        )

        samples = sample(gamma, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(1.5 / 0.5), rtol=0.1)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
