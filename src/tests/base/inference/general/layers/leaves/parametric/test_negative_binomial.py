import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import (
    NegativeBinomial,
    NegativeBinomialLayer,
    ProductNode,
    SumNode,
)
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        negative_binomial_layer = NegativeBinomialLayer(
            scope=Scope([0]), n=3, p=[0.8, 0.3], n_nodes=2
        )
        s1 = SumNode(children=[negative_binomial_layer], weights=[0.3, 0.7])

        negative_binomial_nodes = [
            NegativeBinomial(Scope([0]), n=3, p=0.8),
            NegativeBinomial(Scope([0]), n=3, p=0.3),
        ]
        s2 = SumNode(children=negative_binomial_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [1], [0]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        negative_binomial_layer = NegativeBinomialLayer(
            scope=[Scope([0]), Scope([1])], n=[3, 5], p=[0.8, 0.3]
        )
        p1 = ProductNode(children=[negative_binomial_layer])

        negative_binomial_nodes = [
            NegativeBinomial(Scope([0]), n=3, p=0.8),
            NegativeBinomial(Scope([1]), n=5, p=0.3),
        ]
        p2 = ProductNode(children=negative_binomial_nodes)

        data = np.array([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
