from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.geometric import (
    GeometricLayer,
)
from spflow.base.inference.layers.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        geometric_layer = GeometricLayer(
            scope=Scope([0]), p=[0.8, 0.3], n_nodes=2
        )
        s1 = SPNSumNode(children=[geometric_layer], weights=[0.3, 0.7])

        geometric_nodes = [
            Geometric(Scope([0]), p=0.8),
            Geometric(Scope([0]), p=0.3),
        ]
        s2 = SPNSumNode(children=geometric_nodes, weights=[0.3, 0.7])

        data = np.array([[3], [1], [5]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        geometric_layer = GeometricLayer(
            scope=[Scope([0]), Scope([1])], p=[0.8, 0.3]
        )
        p1 = SPNProductNode(children=[geometric_layer])

        geometric_nodes = [
            Geometric(Scope([0]), p=0.8),
            Geometric(Scope([1]), p=0.3),
        ]
        p2 = SPNProductNode(children=geometric_nodes)

        data = np.array([[3, 1], [2, 7], [5, 4]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
