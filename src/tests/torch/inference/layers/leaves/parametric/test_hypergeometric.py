from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer
from spflow.torch.inference.layers.leaves.parametric.hypergeometric import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.torch.inference.nodes.leaves.parametric.hypergeometric import log_likelihood
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

        layer = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], N=[5, 7, 5], M=[2, 5, 2], n=[3, 2, 3])

        nodes = [
            Hypergeometric(Scope([0]), N=5, M=2, n=3),
            Hypergeometric(Scope([1]), N=7, M=5, n=2),
            Hypergeometric(Scope([0]), N=5, M=2, n=3),
        ]

        dummy_data = torch.tensor([[2, 1], [0, 2], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()