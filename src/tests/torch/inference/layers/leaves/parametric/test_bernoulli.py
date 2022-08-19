from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.bernoulli import BernoulliLayer
from spflow.torch.inference.layers.leaves.parametric.bernoulli import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.torch.inference.nodes.leaves.parametric.bernoulli import log_likelihood
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

        layer = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.5, 0.9])

        nodes = [
            Bernoulli(Scope([0]), p=0.2),
            Bernoulli(Scope([1]), p=0.5),
            Bernoulli(Scope([0]), p=0.9),
        ]

        dummy_data = torch.tensor([[1, 0], [0, 0], [1, 1]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()