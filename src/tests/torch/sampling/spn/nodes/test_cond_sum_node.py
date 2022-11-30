import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import CondSumNode, Gaussian, ProductNode


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_spn_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        s = CondSumNode(
            children=[
                CondSumNode(
                    children=[
                        ProductNode(
                            children=[
                                Gaussian(Scope([0]), -7.0, 1.0),
                                Gaussian(Scope([1]), 7.0, 1.0),
                            ],
                        ),
                        ProductNode(
                            children=[
                                Gaussian(Scope([0]), -5.0, 1.0),
                                Gaussian(Scope([1]), 5.0, 1.0),
                            ],
                        ),
                    ],
                    cond_f=lambda data: {"weights": [0.2, 0.8]},
                ),
                CondSumNode(
                    children=[
                        ProductNode(
                            children=[
                                Gaussian(Scope([0]), -3.0, 1.0),
                                Gaussian(Scope([1]), 3.0, 1.0),
                            ],
                        ),
                        ProductNode(
                            children=[
                                Gaussian(Scope([0]), -1.0, 1.0),
                                Gaussian(Scope([1]), 1.0, 1.0),
                            ],
                        ),
                    ],
                    cond_f=lambda data: {"weights": [0.6, 0.4]},
                ),
            ],
            cond_f=lambda data: {"weights": [0.7, 0.3]},
        )

        samples = sample(s, 1000)
        expected_mean = 0.7 * (0.2 * torch.tensor([-7, 7]) + 0.8 * torch.tensor([-5, 5])) + 0.3 * (
            0.6 * torch.tensor([-3, 3]) + 0.4 * torch.tensor([-1, 1])
        )

        self.assertTrue(torch.allclose(samples.mean(dim=0), expected_mean, rtol=0.1))

    def test_sum_node_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Gaussian(Scope([0]), -5.0, 1.0)
        l2 = Gaussian(Scope([0]), 5.0, 1.0)

        # ----- weights 0, 1 -----

        s = CondSumNode([l1, l2], cond_f=lambda data: {"weights": [0.001, 0.999]})

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(5.0), rtol=0.1))

        # ----- weights 1, 0 -----

        s = CondSumNode([l1, l2], cond_f=lambda data: {"weights": [0.999, 0.001]})

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(-5.0), rtol=0.1))

        # ----- weights 0.2, 0.8 -----

        s = CondSumNode([l1, l2], cond_f=lambda data: {"weights": [0.2, 0.8]})

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(3.0), rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
