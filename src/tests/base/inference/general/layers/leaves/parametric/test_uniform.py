import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import ProductNode, SumNode, Uniform, UniformLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        uniform_layer = UniformLayer(scope=Scope([0]), start=[0.4, 0.3], end=[1.3, 0.8], n_nodes=2)
        s1 = SumNode(children=[uniform_layer], weights=[0.3, 0.7])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([0]), start=0.3, end=0.8),
        ]
        s2 = SumNode(children=uniform_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [0.75], [0.42]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        uniform_layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.4, 0.3], end=[1.3, 0.8])
        p1 = ProductNode(children=[uniform_layer])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([1]), start=0.3, end=0.8),
        ]
        p2 = ProductNode(children=uniform_nodes)

        data = np.array([[0.5, 0.53], [0.42, 0.6], [0.47, 0.7]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
