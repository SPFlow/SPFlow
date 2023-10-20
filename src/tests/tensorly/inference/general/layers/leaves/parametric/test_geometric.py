import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.meta.data import Scope
from spflow.tensorly.structure.general.nodes.leaves import Geometric
from spflow.tensorly.structure.general.layers.leaves import GeometricLayer


class TestNode(unittest.TestCase):
    def test_layer_likelihood_1(self):

        geometric_layer = GeometricLayer(scope=Scope([0]), p=[0.8, 0.3], n_nodes=2)
        s1 = SumNode(children=[geometric_layer], weights=[0.3, 0.7])

        geometric_nodes = [
            Geometric(Scope([0]), p=0.8),
            Geometric(Scope([0]), p=0.3),
        ]
        s2 = SumNode(children=geometric_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[3], [1], [5]])

        self.assertTrue(tl_allclose(log_likelihood(s1, data), log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        geometric_layer = GeometricLayer(scope=[Scope([0]), Scope([1])], p=[0.8, 0.3])
        p1 = ProductNode(children=[geometric_layer])

        geometric_nodes = [
            Geometric(Scope([0]), p=0.8),
            Geometric(Scope([1]), p=0.3),
        ]
        p2 = ProductNode(children=geometric_nodes)

        data = tl.tensor([[3, 1], [2, 7], [5, 4]])

        self.assertTrue(tl_allclose(log_likelihood(p1, data), log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()