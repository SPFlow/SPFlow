from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.gaussian import GaussianLayer
from spflow.base.inference.layers.leaves.parametric.gaussian import log_likelihood
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.inference.nodes.leaves.parametric.gaussian import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        gaussian_layer = GaussianLayer(scope=Scope([0]), mean=[0.8, 0.3], std=[1.3, 0.4], n_nodes=2)
        s1 = SPNSumNode(children=[gaussian_layer], weights=[0.3, 0.7])

        gaussian_nodes = [Gaussian(Scope([0]), mean=0.8, std=1.3), Gaussian(Scope([0]), mean=0.3, std=0.4)]
        s2 = SPNSumNode(children=gaussian_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))
    
    def test_layer_likelihood_2(self):

        gaussian_layer = GaussianLayer(scope=[Scope([0]), Scope([1])], mean=[0.8, 0.3], std=[1.3, 0.4])
        p1 = SPNProductNode(children=[gaussian_layer])

        gaussian_nodes = [Gaussian(Scope([0]), mean=0.8, std=1.3), Gaussian(Scope([1]), mean=0.3, std=0.4)]
        p2 = SPNProductNode(children=gaussian_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()