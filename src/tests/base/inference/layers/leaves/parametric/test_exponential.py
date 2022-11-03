from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.exponential import (
    ExponentialLayer,
)
from spflow.base.inference.layers.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential,
)
from spflow.base.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.base.structure.spn.nodes.sum_node import SPNSumNode
from spflow.base.inference.spn.nodes.sum_node import log_likelihood
from spflow.base.structure.spn.nodes.product_node import SPNProductNode
from spflow.base.inference.spn.nodes.product_node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        exponential_layer = ExponentialLayer(
            scope=Scope([0]), l=[0.8, 0.3], n_nodes=2
        )
        s1 = SPNSumNode(children=[exponential_layer], weights=[0.3, 0.7])

        exponential_nodes = [
            Exponential(Scope([0]), l=0.8),
            Exponential(Scope([0]), l=0.3),
        ]
        s2 = SPNSumNode(children=exponential_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        exponential_layer = ExponentialLayer(
            scope=[Scope([0]), Scope([1])], l=[0.8, 0.3]
        )
        p1 = SPNProductNode(children=[exponential_layer])

        exponential_nodes = [
            Exponential(Scope([0]), l=0.8),
            Exponential(Scope([1]), l=0.3),
        ]
        p2 = SPNProductNode(children=exponential_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
