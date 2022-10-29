from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.uniform import UniformLayer
from spflow.base.inference.layers.leaves.parametric.uniform import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.inference.nodes.leaves.parametric.uniform import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        uniform_layer = UniformLayer(
            scope=Scope([0]), start=[0.4, 0.3], end=[1.3, 0.8], n_nodes=2
        )
        s1 = SPNSumNode(children=[uniform_layer], weights=[0.3, 0.7])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([0]), start=0.3, end=0.8),
        ]
        s2 = SPNSumNode(children=uniform_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [0.75], [0.42]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        uniform_layer = UniformLayer(
            scope=[Scope([0]), Scope([1])], start=[0.4, 0.3], end=[1.3, 0.8]
        )
        p1 = SPNProductNode(children=[uniform_layer])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([1]), start=0.3, end=0.8),
        ]
        p2 = SPNProductNode(children=uniform_nodes)

        data = np.array([[0.5, 0.53], [0.42, 0.6], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
