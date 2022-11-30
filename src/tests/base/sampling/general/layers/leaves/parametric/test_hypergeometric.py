import random
import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.sampling import sample
from spflow.base.structure.spn import (
    Hypergeometric,
    HypergeometricLayer,
    ProductNode,
    SumNode,
)
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        hypergeometric_layer = HypergeometricLayer(
            scope=Scope([0]), N=8, M=3, n=4, n_nodes=2
        )
        s1 = SumNode(children=[hypergeometric_layer], weights=[0.3, 0.7])

        hypergeometric_nodes = [
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
        ]
        s2 = SumNode(children=hypergeometric_nodes, weights=[0.3, 0.7])

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

        hypergeometric_layer = HypergeometricLayer(
            scope=[Scope([0]), Scope([1])], N=[8, 10], M=[3, 2], n=[4, 5]
        )
        p1 = ProductNode(children=[hypergeometric_layer])

        hypergeometric_nodes = [
            Hypergeometric(Scope([0]), N=8, M=3, n=4),
            Hypergeometric(Scope([1]), N=10, M=2, n=5),
        ]
        p2 = ProductNode(children=hypergeometric_nodes)

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

        hypergeometric_layer = HypergeometricLayer(
            scope=Scope([0]), N=8, M=3, n=4, n_nodes=2
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, hypergeometric_layer)


if __name__ == "__main__":
    unittest.main()
