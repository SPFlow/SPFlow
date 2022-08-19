from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.binomial import BinomialLayer
from spflow.torch.inference.layers.leaves.parametric.binomial import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.torch.inference.nodes.leaves.parametric.binomial import log_likelihood
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

        layer = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], n=[3, 2, 3], p=[0.2, 0.5, 0.9])

        nodes = [
            Binomial(Scope([0]), n=3, p=0.2),
            Binomial(Scope([1]), n=2, p=0.5),
            Binomial(Scope([0]), n=3, p=0.9),
        ]

        dummy_data = torch.tensor([[3, 1], [1, 2], [0, 0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()