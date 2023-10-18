import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import Categorical, CategoricalLayer, ProductNode, SumNode
from spflow.meta.data import Scope


class TestCategorical(unittest.TestCase):
    def test_layer_likelihood_1(self):

        layer = CategoricalLayer(scope=Scope([0]), k = [2, 2], p = [[0.5, 0.5], [0.3, 0.7]], n_nodes=2)
        s1 = SumNode(children=[layer], weights=[0.3, 0.7])

        categorical_nodes = [
            Categorical(Scope([0]), k=2, p=[0.5, 0.5]),
            Categorical(Scope([0]), k=2, p=[0.3, 0.7])
        ]
        s2 = SumNode(children=categorical_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [1], [0]])
        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))


    def test_layer_likelihood_2(self):

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])], k = [2, 2], p = [[0.5, 0.5], [0.3, 0.7]], n_nodes=2)
        p1 = ProductNode(children=[layer])

        categorical_nodes = [
            Categorical(Scope([0]), k=2, p=[0.5, 0.5]),
            Categorical(Scope([1]), k=2, p=[0.3, 0.7])
        ]
        p2 = ProductNode(children=categorical_nodes)

        data = np.array([[0, 1], [1, 1], [0, 0]])
        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()