from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer
from spflow.torch.inference.layers.leaves.parametric.uniform import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.torch.inference.nodes.leaves.parametric.uniform import log_likelihood
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

        layer = UniformLayer(scope=[Scope([0]), Scope([1]), Scope([0])], start=[0.2, -1.0, 0.3], end=[1.0, 0.3, 0.97])

        nodes = [
            Uniform(Scope([0]), start=0.2, end=1.0, support_outside=True),
            Uniform(Scope([1]), start=-1.0, end=0.3, support_outside=True),
            Uniform(Scope([0]), start=0.3, end=0.97, support_outside=True),
        ]

        dummy_data = torch.tensor([[0.5, -0.3], [0.9, 0.21], [0.5, 0.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()