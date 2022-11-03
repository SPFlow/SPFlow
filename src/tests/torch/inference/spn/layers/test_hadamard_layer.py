from spflow.meta.data import Scope
from spflow.torch.structure.spn import (
    SumNode,
    ProductNode,
    Gaussian,
    HadamardLayer,
)
from spflow.torch.inference import log_likelihood

import torch
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_hadamard_layer_likelihood(self):

        input_partitions = [
            [Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
            [Gaussian(Scope([3])), Gaussian(Scope([3])), Gaussian(Scope([3]))],
        ]

        layer_spn = SumNode(
            children=[HadamardLayer(child_partitions=input_partitions)],
            weights=[0.3, 0.2, 0.5],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        input_partitions[0][i],
                        input_partitions[1][j],
                        input_partitions[2][k],
                        input_partitions[3][l],
                    ]
                )
                for (i, j, k, l) in [[0, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 2]]
            ],
            weights=[0.3, 0.2, 0.5],
        )

        dummy_data = torch.tensor(
            [
                [1.0, 0.25, 0.0, -0.7],
                [0.0, 1.0, 0.25, 0.12],
                [0.25, 0.0, 1.0, 0.0],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
