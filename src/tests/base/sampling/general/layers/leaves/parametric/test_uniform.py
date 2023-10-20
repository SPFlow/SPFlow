import random
import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.sampling import sample
from spflow.base.structure.spn import ProductNode, SumNode, Uniform, UniformLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        uniform_layer = UniformLayer(scope=Scope([0]), start=[0.4, 0.3], end=[1.3, 0.8], n_nodes=2)
        s1 = SumNode(children=[uniform_layer], weights=[0.3, 0.7])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([0]), start=0.3, end=0.8),
        ]
        s2 = SumNode(children=uniform_nodes, weights=[0.3, 0.7])

        layer_samples = sample(s1, 10000)
        nodes_samples = sample(s2, 10000)
        self.assertTrue(
            np.allclose(
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

        uniform_layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.4, 0.3], end=[1.3, 0.8])
        p1 = ProductNode(children=[uniform_layer])

        uniform_nodes = [
            Uniform(Scope([0]), start=0.4, end=1.3),
            Uniform(Scope([1]), start=0.3, end=0.8),
        ]
        p2 = ProductNode(children=uniform_nodes)

        layer_samples = sample(p1, 10000)
        nodes_samples = sample(p2, 10000)
        self.assertTrue(
            np.allclose(
                layer_samples.mean(axis=0),
                nodes_samples.mean(axis=0),
                atol=0.01,
                rtol=0.1,
            )
        )

    def test_sampling_3(self):

        uniform_layer = UniformLayer(scope=Scope([0]), start=[0.4, 0.3], end=[1.3, 0.8], n_nodes=2)

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, uniform_layer)


if __name__ == "__main__":
    unittest.main()