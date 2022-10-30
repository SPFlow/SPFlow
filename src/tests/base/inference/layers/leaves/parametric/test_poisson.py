from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.poisson import PoissonLayer
from spflow.base.inference.layers.leaves.parametric.poisson import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.base.inference.nodes.leaves.parametric.poisson import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        poisson_layer = PoissonLayer(scope=Scope([0]), l=[0.8, 0.3], n_nodes=2)
        s1 = SPNSumNode(children=[poisson_layer], weights=[0.3, 0.7])

        poisson_nodes = [Poisson(Scope([0]), l=0.8), Poisson(Scope([0]), l=0.3)]
        s2 = SPNSumNode(children=poisson_nodes, weights=[0.3, 0.7])

        data = np.array([[1], [5], [3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        poisson_layer = PoissonLayer(
            scope=[Scope([0]), Scope([1])], l=[0.8, 0.3]
        )
        p1 = SPNProductNode(children=[poisson_layer])

        poisson_nodes = [Poisson(Scope([0]), l=0.8), Poisson(Scope([1]), l=0.3)]
        p2 = SPNProductNode(children=poisson_nodes)

        data = np.array([[1, 6], [5, 3], [3, 7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
