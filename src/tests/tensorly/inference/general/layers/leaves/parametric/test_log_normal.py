import unittest

import tensorly as tl

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.meta.data import Scope
from spflow.tensorly.structure.general.nodes.leaves import LogNormal
from spflow.tensorly.structure.general.layers.leaves import LogNormalLayer


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        log_normal_layer = LogNormalLayer(scope=Scope([0]), mean=[0.8, 0.3], std=[1.3, 0.4], n_nodes=2)
        s1 = SumNode(children=[log_normal_layer], weights=[0.3, 0.7])

        log_normal_nodes = [
            LogNormal(Scope([0]), mean=0.8, std=1.3),
            LogNormal(Scope([0]), mean=0.3, std=0.4),
        ]
        s2 = SumNode(children=log_normal_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0.5], [1.5], [0.3]])

        self.assertTrue(tl.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        log_normal_layer = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=[0.8, 0.3], std=[1.3, 0.4])
        p1 = ProductNode(children=[log_normal_layer])

        log_normal_nodes = [
            LogNormal(Scope([0]), mean=0.8, std=1.3),
            LogNormal(Scope([1]), mean=0.3, std=0.4),
        ]
        p2 = ProductNode(children=log_normal_nodes)

        data = tl.tensor([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(tl.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
