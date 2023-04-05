import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.meta.data import Scope
from spflow.tensorly.structure.general.nodes.leaves import Poisson
from spflow.tensorly.structure.general.layers.leaves import PoissonLayer


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        poisson_layer = PoissonLayer(scope=Scope([0]), l=[0.8, 0.3], n_nodes=2)
        s1 = SumNode(children=[poisson_layer], weights=[0.3, 0.7])

        poisson_nodes = [Poisson(Scope([0]), l=0.8), Poisson(Scope([0]), l=0.3)]
        s2 = SumNode(children=poisson_nodes, weights=[0.3, 0.7])

        layer_samples = sample(s1, 10000)
        nodes_samples = sample(s2, 10000)
        self.assertTrue(
            tl_allclose(
                layer_samples.mean(axis=0),
                nodes_samples.mean(axis=0),
                atol=0.01,
                rtol=0.1,
            )
        )

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        poisson_layer = PoissonLayer(scope=[Scope([0]), Scope([1])], l=[0.8, 0.3])
        p1 = ProductNode(children=[poisson_layer])

        poisson_nodes = [Poisson(Scope([0]), l=0.8), Poisson(Scope([1]), l=0.3)]
        p2 = ProductNode(children=poisson_nodes)

        layer_samples = sample(p1, 10000)
        nodes_samples = sample(p2, 10000)
        self.assertTrue(
            tl_allclose(
                layer_samples.mean(axis=0),
                nodes_samples.mean(axis=0),
                atol=0.01,
                rtol=0.1,
            )
        )

    def test_sampling_3(self):

        poisson_layer = PoissonLayer(scope=Scope([0]), l=[0.8, 0.3], n_nodes=2)

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, poisson_layer)


if __name__ == "__main__":
    unittest.main()
