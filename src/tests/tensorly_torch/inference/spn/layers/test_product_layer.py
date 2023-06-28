import unittest

import torch

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import Gaussian, ProductLayer, ProductNode, SumNode
from spflow.torch.structure.general.nodes.leaves import Gaussian
from spflow.tensorly.structure.spn.nodes.sum_node import toLayerBased

class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_product_layer_likelihood(self):

        input_nodes = [
            Gaussian(Scope([0])),
            Gaussian(Scope([1])),
            Gaussian(Scope([2])),
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
