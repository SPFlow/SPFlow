from ..structure.dummy_module import DummyNestedModule
from ..structure.nodes.dummy_node import DummyNode
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.inference.module import likelihood, log_likelihood
import numpy as np
import unittest


class TestModule(unittest.TestCase):
    def test_likelihood(self):

        dummy_nodes = [DummyNode(Scope([0]))]
        print(type(dummy_nodes[0]), isinstance(dummy_nodes[0], Module))

        dummy_module = DummyNestedModule(children=dummy_nodes)

        dummy_data = np.array([[np.nan, 0.0, 1.0]])

        self.assertRaises(LookupError, log_likelihood, dummy_module.placeholders[0], dummy_data)
        self.assertRaises(LookupError, likelihood, dummy_module.placeholders[0], dummy_data)


if __name__ == "__main__":
    unittest.main()