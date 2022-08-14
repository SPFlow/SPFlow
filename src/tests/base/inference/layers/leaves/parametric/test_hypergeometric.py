from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer
from spflow.base.inference.layers.leaves.parametric.hypergeometric import log_likelihood
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.base.inference.nodes.leaves.parametric.hypergeometric import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        hypergeometric_layer = HypergeometricLayer(scope=Scope([0]), N=[8, 10], M=[3, 2], n=[4, 5], n_nodes=2)
        s1 = SPNSumNode(children=[hypergeometric_layer], weights=[0.3, 0.7])

        hypergeometric_nodes = [Hypergeometric(Scope([0]), N=8, M=3, n=4), Hypergeometric(Scope([0]), N=10, M=2, n=5)]
        s2 = SPNSumNode(children=hypergeometric_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [2], [1]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))
    
    def test_layer_likelihood_2(self):

        hypergeometric_layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[8, 10], M=[3, 2], n=[4, 5])
        p1 = SPNProductNode(children=[hypergeometric_layer])

        hypergeometric_nodes = [Hypergeometric(Scope([0]), N=8, M=3, n=4), Hypergeometric(Scope([1]), N=10, M=2, n=5)]
        p2 = SPNProductNode(children=hypergeometric_nodes)

        data = np.array([[2, 0], [3, 1], [0, 2]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()