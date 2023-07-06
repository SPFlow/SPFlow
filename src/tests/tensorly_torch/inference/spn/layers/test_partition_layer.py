import itertools
import unittest

import torch

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import Gaussian, PartitionLayer, ProductNode, SumNode
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn.nodes.sum_node import toLayerBased

class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_partition_layer_likelihood(self):

        input_partitions = [
            [Gaussian(Scope([0])), Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
        ]

        layer_spn = SumNode(
            children=[PartitionLayer(child_partitions=input_partitions)],
            weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        input_partitions[0][i],
                        input_partitions[1][j],
                        input_partitions[2][k],
                    ]
                )
                for (i, j, k) in itertools.product([0, 1], [0, 1, 2], [0])
            ],
            weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
        )
        layer_based_spn = toLayerBased(layer_spn)

        dummy_data = torch.tensor([[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.25, 0.0, 1.0]])

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)
        lb_ll = log_likelihood(layer_based_spn, dummy_data)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))
        self.assertTrue(torch.allclose(layer_ll, torch.tensor(lb_ll, dtype=float)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
