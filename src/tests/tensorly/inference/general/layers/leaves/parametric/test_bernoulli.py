import unittest

import tensorly as tl

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import Bernoulli, BernoulliLayer, ProductNode, SumNode
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        bernoulli_layer = BernoulliLayer(scope=Scope([0]), p=[0.8, 0.3], n_nodes=2)
        s1 = SumNode(children=[bernoulli_layer], weights=[0.3, 0.7])

        bernoulli_nodes = [
            Bernoulli(Scope([0]), p=0.8),
            Bernoulli(Scope([0]), p=0.3),
        ]
        s2 = SumNode(children=bernoulli_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0], [1], [0]])

        self.assertTrue(tl.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        bernoulli_layer = BernoulliLayer(scope=[Scope([0]), Scope([1])], p=[0.8, 0.3])
        p1 = ProductNode(children=[bernoulli_layer])

        bernoulli_nodes = [
            Bernoulli(Scope([0]), p=0.8),
            Bernoulli(Scope([1]), p=0.3),
        ]
        p2 = ProductNode(children=bernoulli_nodes)

        data = tl.tensor([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(tl.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
