import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.meta.data import Scope
from spflow.tensorly.structure.general.nodes.leaves import Gaussian
from spflow.tensorly.structure.general.layers.leaves import GaussianLayer


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        gaussian_layer = GaussianLayer(scope=Scope([0]), mean=[0.8, 0.3], std=[1.3, 0.4], n_nodes=2)
        s1 = SumNode(children=[gaussian_layer], weights=[0.3, 0.7])

        gaussian_nodes = [
            Gaussian(Scope([0]), mean=0.8, std=1.3),
            Gaussian(Scope([0]), mean=0.3, std=0.4),
        ]
        s2 = SumNode(children=gaussian_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0.5], [1.5], [0.3]])

        self.assertTrue(tl_allclose(log_likelihood(s1, data), log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        gaussian_layer = GaussianLayer(scope=[Scope([0]), Scope([1])], mean=[0.8, 0.3], std=[1.3, 0.4])
        p1 = ProductNode(children=[gaussian_layer])

        gaussian_nodes = [
            Gaussian(Scope([0]), mean=0.8, std=1.3),
            Gaussian(Scope([1]), mean=0.3, std=0.4),
        ]
        p2 = ProductNode(children=gaussian_nodes)

        data = tl.tensor([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(tl_allclose(log_likelihood(p1, data), log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
