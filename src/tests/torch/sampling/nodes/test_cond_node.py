from spflow.meta.data.scope import Scope
from spflow.torch.structure.nodes.cond_node import SPNCondSumNode
from spflow.torch.structure.nodes.node import SPNProductNode
from spflow.torch.sampling.nodes.cond_node import sample
from spflow.torch.sampling.nodes.node import sample
from spflow.torch.inference.nodes.cond_node import log_likelihood
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.sampling.nodes.leaves.parametric.gaussian import sample
from spflow.torch.sampling.module import sample

import torch
import numpy as np
import random
import unittest


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

        s = SPNCondSumNode(
            children=[
                SPNCondSumNode(
                    children=[
                        SPNProductNode(
                            children=[
                                Gaussian(Scope([0]), -7.0, 1.0),
                                Gaussian(Scope([1]), 7.0, 1.0),
                            ],
                        ),
                        SPNProductNode(
                            children=[
                                Gaussian(Scope([0]), -5.0, 1.0),
                                Gaussian(Scope([1]), 5.0, 1.0),
                            ],
                        ),
                    ],
                    cond_f=lambda data: {"weights": [0.2, 0.8]},
                ),
                SPNCondSumNode(
                    children=[
                        SPNProductNode(
                            children=[
                                Gaussian(Scope([0]), -3.0, 1.0),
                                Gaussian(Scope([1]), 3.0, 1.0),
                            ],
                        ),
                        SPNProductNode(
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
        expected_mean = 0.7 * (
            0.2 * torch.tensor([-7, 7]) + 0.8 * torch.tensor([-5, 5])
        ) + 0.3 * (0.6 * torch.tensor([-3, 3]) + 0.4 * torch.tensor([-1, 1]))

        self.assertTrue(
            torch.allclose(samples.mean(dim=0), expected_mean, rtol=0.1)
        )

    def test_sum_node_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Gaussian(Scope([0]), -5.0, 1.0)
        l2 = Gaussian(Scope([0]), 5.0, 1.0)

        # ----- weights 0, 1 -----

        s = SPNCondSumNode(
            [l1, l2], cond_f=lambda data: {"weights": [0.001, 0.999]}
        )

        samples = sample(s, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(5.0), rtol=0.1)
        )

        # ----- weights 1, 0 -----

        s = SPNCondSumNode(
            [l1, l2], cond_f=lambda data: {"weights": [0.999, 0.001]}
        )

        samples = sample(s, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(-5.0), rtol=0.1)
        )

        # ----- weights 0.2, 0.8 -----

        s = SPNCondSumNode(
            [l1, l2], cond_f=lambda data: {"weights": [0.2, 0.8]}
        )

        samples = sample(s, 1000)
        self.assertTrue(
            torch.isclose(samples.mean(), torch.tensor(3.0), rtol=0.1)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
