import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.inference import log_likelihood
from spflow.tensorly.sampling import sample
from spflow.torch.structure.spn import Gaussian
from spflow.tensorly.structure.spn import SumLayer, SumNode
from spflow.tensorly.structure.spn.nodes.sum_node import toLayerBased


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sum_layer_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        input_nodes = [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
            Gaussian(Scope([0]), mean=0.0, std=0.01),
        ]

        layer_spn = SumNode(
            children=[
                SumLayer(
                    n_nodes=3,
                    children=input_nodes,
                    weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
                ),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        nodes_spn = SumNode(
            children=[
                SumNode(children=input_nodes, weights=[0.8, 0.1, 0.1]),
                SumNode(children=input_nodes, weights=[0.2, 0.3, 0.5]),
                SumNode(children=input_nodes, weights=[0.2, 0.7, 0.1]),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        layerbased_spn = toLayerBased(layer_spn)

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)
        layerbased_samples = sample(layerbased_spn, 10000)

        expected_mean = (
            0.3 * (0.8 * 3.0 + 0.1 * 1.0 + 0.1 * 0.0)
            + 0.4 * (0.2 * 3.0 + 0.3 * 1.0 + 0.5 * 0.0)
            + 0.3 * (0.2 * 3.0 + 0.7 * 1.0 + 0.1 * 0.0)
        )
        self.assertTrue(
            torch.allclose(
                nodes_samples.mean(dim=0),
                torch.tensor([expected_mean]),
                atol=0.01,
                rtol=0.1,
            )
        )
        self.assertTrue(
            torch.allclose(
                layer_samples.mean(dim=0),
                nodes_samples.mean(dim=0),
                atol=0.01,
                rtol=0.1,
            )
        )

        # sample from multiple outputs (with same scope)
        self.assertRaises(
            ValueError,
            sample,
            list(layer_spn.children)[0],
            1,
            sampling_ctx=SamplingContext([0], [[0, 1]]),
        )

        self.assertTrue(
            torch.allclose(
                layer_samples.mean(dim=0),
                layerbased_samples.mean(dim=0),
                atol=0.01,
                rtol=0.1,
            )
        )

        # sample from multiple outputs (with same scope)
        self.assertRaises(
            ValueError,
            sample,
            list(layerbased_spn.children)[0],
            1,
            sampling_ctx=SamplingContext([0], [[0, 1]]),
        )



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
