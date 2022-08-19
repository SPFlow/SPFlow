from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.poisson import PoissonLayer
from spflow.torch.inference.layers.leaves.parametric.poisson import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.torch.inference.nodes.leaves.parametric.poisson import log_likelihood
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

        layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

        nodes = [
            Poisson(Scope([0]), l=0.2),
            Poisson(Scope([1]), l=1.0),
            Poisson(Scope([0]), l=2.3),
        ]

        dummy_data = torch.tensor([[1, 3], [3, 7], [2, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()