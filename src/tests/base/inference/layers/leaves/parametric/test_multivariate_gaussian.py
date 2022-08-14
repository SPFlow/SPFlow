from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer
from spflow.base.inference.layers.leaves.parametric.multivariate_gaussian import log_likelihood
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        multivariate_gaussian_layer = MultivariateGaussianLayer(scope=Scope([0,1]), mean=[[0.8, 0.3], [0.2, -0.1]], cov=[[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]], n_nodes=2)
        s1 = SPNSumNode(children=[multivariate_gaussian_layer], weights=[0.3, 0.7])

        multivariate_gaussian_nodes = [MultivariateGaussian(Scope([0,1]), mean=[0.8, 0.3], cov=[[1.3, 0.4], [0.4, 0.5]]), MultivariateGaussian(Scope([0,1]), mean=[0.2, -0.1], cov=[[0.5, 0.1], [0.1, 1.4]])]
        s2 = SPNSumNode(children=multivariate_gaussian_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5, 0.3], [1.5, -0.3], [0.3, 0.0]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))
    
    def test_layer_likelihood_2(self):

        multivariate_gaussian_layer = MultivariateGaussianLayer(scope=[Scope([0,1]), Scope([2,3])], mean=[[0.8, 0.3], [0.2, -0.1]], cov=[[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]])
        p1 = SPNProductNode(children=[multivariate_gaussian_layer])

        multivariate_gaussian_nodes = [MultivariateGaussian(Scope([0,1]), mean=[0.8, 0.3], cov=[[1.3, 0.4], [0.4, 0.5]]), MultivariateGaussian(Scope([2,3]), mean=[0.2, -0.1], cov=[[0.5, 0.1], [0.1, 1.4]])]
        p2 = SPNProductNode(children=multivariate_gaussian_nodes)

        data = np.array([[0.5, 1.6, -0.1, 3.0], [0.1, 0.3, 0.9, 0.73], [0.47, 0.7, 0.5, 0.1]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()