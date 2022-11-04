from spflow.meta.data import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.torch.structure.spn.layers.cond_sum_layer import CondSumLayer
from spflow.torch.inference.spn.layers.cond_sum_layer import log_likelihood
from spflow.torch.structure.spn.nodes.cond_sum_node import CondSumNode
from spflow.torch.inference.spn.nodes.cond_sum_node import log_likelihood
from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.general.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.inference.module import log_likelihood
import torch
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sum_layer_likelihood(self):

        input_nodes = [
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ]

        layer_spn = CondSumNode(
            children=[
                CondSumLayer(
                    n_nodes=3,
                    children=input_nodes,
                    cond_f=lambda data: {
                        "weights": [
                            [0.8, 0.1, 0.1],
                            [0.2, 0.3, 0.5],
                            [0.2, 0.7, 0.1],
                        ]
                    },
                ),
            ],
            cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
        )

        nodes_spn = CondSumNode(
            children=[
                CondSumNode(
                    children=input_nodes,
                    cond_f=lambda data: {"weights": [0.8, 0.1, 0.1]},
                ),
                CondSumNode(
                    children=input_nodes,
                    cond_f=lambda data: {"weights": [0.2, 0.3, 0.5]},
                ),
                CondSumNode(
                    children=input_nodes,
                    cond_f=lambda data: {"weights": [0.2, 0.7, 0.1]},
                ),
            ],
            cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
        )

        dummy_data = torch.tensor(
            [
                [1.0],
                [
                    0.0,
                ],
                [0.25],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_sum_layer_gradient_computation(self):

        torch.manual_seed(0)

        # generate random weights for a sum node with two children
        weights = torch.tensor(
            [[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], requires_grad=True
        )

        data_1 = torch.randn((70000, 1))
        data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
        data_2 = torch.randn((30000, 1))
        data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

        data = torch.cat([data_1, data_2])

        # initialize Gaussians
        gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
        gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

        # sum layer to be optimized
        sum_layer = CondSumLayer(
            n_nodes=3,
            children=[gaussian_1, gaussian_2],
            cond_f=lambda data: {"weights": weights},
        )

        ll = log_likelihood(sum_layer, data).mean()
        ll.backward()

        self.assertTrue(weights.grad is not None)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
