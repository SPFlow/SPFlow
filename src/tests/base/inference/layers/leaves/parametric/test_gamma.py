from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.gamma import GammaLayer
from spflow.base.inference.layers.leaves.parametric.gamma import log_likelihood
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.inference.nodes.leaves.parametric.gamma import log_likelihood
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        gamma_layer = GammaLayer(
            scope=Scope([0]), alpha=[0.8, 0.3], beta=[1.3, 0.4], n_nodes=2
        )
        s1 = SPNSumNode(children=[gamma_layer], weights=[0.3, 0.7])

        gamma_nodes = [
            Gamma(Scope([0]), alpha=0.8, beta=1.3),
            Gamma(Scope([0]), alpha=0.3, beta=0.4),
        ]
        s2 = SPNSumNode(children=gamma_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        gamma_layer = GammaLayer(
            scope=[Scope([0]), Scope([1])], alpha=[0.8, 0.3], beta=[1.3, 0.4]
        )
        p1 = SPNProductNode(children=[gamma_layer])

        gamma_nodes = [
            Gamma(Scope([0]), alpha=0.8, beta=1.3),
            Gamma(Scope([1]), alpha=0.3, beta=0.4),
        ]
        p2 = SPNProductNode(children=gamma_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
