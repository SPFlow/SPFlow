import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.inference import log_likelihood
from spflow.torch.sampling import sample
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussianLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CondMultivariateGaussianLayer(
            scope=[Scope([0, 1], [4]), Scope([2, 3], [4]), Scope([0, 1], [4])],
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1], [0.8, 0.3]],
                "cov": [
                    [[0.13, 0.08], [0.08, 0.05]],
                    [[0.17, 0.054], [0.054, 0.0296]],
                    [[0.13, 0.08], [0.08, 0.05]],
                ],
            },
        )

        nodes = [
            CondMultivariateGaussian(
                Scope([0, 1], [4]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[0.13, 0.08], [0.08, 0.05]],
                },
            ),
            CondMultivariateGaussian(
                Scope([2, 3], [4]),
                cond_f=lambda data: {
                    "mean": [0.2, -0.1],
                    "cov": [[0.17, 0.054], [0.054, 0.0296]],
                },
            ),
            CondMultivariateGaussian(
                Scope([0, 1], [4]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[0.13, 0.08], [0.08, 0.05]],
                },
            ),
        ]

        # make sure sampling fron non-overlapping scopes works
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[0, 1]]))
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[2, 1]]))
        # make sure sampling from overlapping scopes does not works
        self.assertRaises(
            ValueError,
            sample,
            layer,
            1,
            sampling_ctx=SamplingContext([0], [[0, 2]]),
        )
        self.assertRaises(
            ValueError,
            sample,
            layer,
            1,
            sampling_ctx=SamplingContext([0], [[]]),
        )

        layer_samples = sample(
            layer,
            10000,
            sampling_ctx=SamplingContext(
                list(range(10000)),
                [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
            ),
        )
        nodes_samples = torch.hstack(
            [
                torch.cat([sample(nodes[0], 5000), sample(nodes[2], 5000)], dim=0),
                sample(nodes[1], 10000)[:, [2, 3]],
            ]
        )

        expected_mean = torch.tensor([0.8, 0.3, 0.2, -0.1])
        self.assertTrue(torch.allclose(nodes_samples.mean(dim=0), expected_mean, atol=0.01, rtol=0.1))
        self.assertTrue(
            torch.allclose(
                layer_samples.mean(dim=0),
                nodes_samples.mean(dim=0),
                atol=0.01,
                rtol=0.1,
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
