import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import (
    ProductNode,
    SumNode,
)
from spflow.tensorly.structure.general.nodes.leaves import Hypergeometric
from spflow.tensorly.structure.general.layers.leaves import HypergeometricLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        hypergeometric_layer = HypergeometricLayer(scope=Scope([0]), N=8, M=3, n=4, n_nodes=2)
        s1 = SumNode(children=[hypergeometric_layer], weights=[0.3, 0.7])

        hypergeometric_nodes = [
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
        ]
        s2 = SumNode(children=hypergeometric_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0], [2], [1]])

        self.assertTrue(tl_allclose(log_likelihood(s1, data), log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        hypergeometric_layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[8, 10], M=[3, 2], n=[4, 5])
        p1 = ProductNode(children=[hypergeometric_layer])

        hypergeometric_nodes = [
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
            Hypergeometric(Scope([1]), N=10, M=2, n=5),
        ]
        p2 = ProductNode(children=hypergeometric_nodes)

        data = tl.tensor([[2, 0], [3, 1], [0, 2]])

        self.assertTrue(tl_allclose(log_likelihood(p1, data), log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
