import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.inference import log_likelihood
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import Gaussian, ProductLayer, ProductNode, SumNode


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_product_layer_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        input_nodes = [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([2]), mean=0.0, std=0.01),
        ]

        layer_spn = SumNode(
            children=[ProductLayer(n_nodes=3, children=input_nodes)],
            weights=[0.3, 0.4, 0.3],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(children=input_nodes),
                ProductNode(children=input_nodes),
                ProductNode(children=input_nodes),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        self.assertTrue(
            torch.allclose(
                nodes_samples.mean(axis=0),
                torch.tensor([3.0, 1.0, 0.0]),
                atol=0.01,
                rtol=0.1,
            )
        )
        self.assertTrue(
            torch.allclose(
                layer_samples.mean(axis=0),
                nodes_samples.mean(axis=0),
                atol=0.01,
                rtol=0.1,
            )
        )

        # sample from multiple outputs (with same scope)
        self.assertRaises(
            ValueError,
            sample,
            list(layer_spn.chs)[0],
            1,
            sampling_ctx=SamplingContext([0], [[0, 1]]),
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
