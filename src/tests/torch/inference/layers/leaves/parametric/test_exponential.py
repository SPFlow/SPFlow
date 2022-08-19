from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.exponential import ExponentialLayer
from spflow.torch.inference.layers.leaves.parametric.exponential import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.exponential import Exponential
from spflow.torch.inference.nodes.leaves.parametric.exponential import log_likelihood
from spflow.torch.inference.module import log_likelihood
import torch
import unittest
import itertools


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

        nodes = [
            Exponential(Scope([0]), l=0.2),
            Exponential(Scope([1]), l=1.0),
            Exponential(Scope([0]), l=2.3),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()