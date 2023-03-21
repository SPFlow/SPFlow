import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import CondGamma, CondGammaLayer, ProductNode, SumNode
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        gamma_layer = CondGammaLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"alpha": [0.8, 0.3], "beta": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SumNode(children=[gamma_layer], weights=[0.3, 0.7])

        gamma_nodes = [
            CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 0.8, "beta": 1.3}),
            CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 0.3, "beta": 0.4}),
        ]
        s2 = SumNode(children=gamma_nodes, weights=[0.3, 0.7])

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

        gamma_layer = CondGammaLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"alpha": [0.8, 0.3], "beta": [1.3, 0.4]},
        )
        p1 = ProductNode(children=[gamma_layer])

        gamma_nodes = [
            CondGamma(Scope([0], [2]), cond_f=lambda data: {"alpha": 0.8, "beta": 1.3}),
            CondGamma(Scope([1], [2]), cond_f=lambda data: {"alpha": 0.3, "beta": 0.4}),
        ]
        p2 = ProductNode(children=gamma_nodes)

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

        gamma_layer = CondGammaLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"alpha": [0.8, 0.3], "beta": [1.3, 0.4]},
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, gamma_layer)


if __name__ == "__main__":
    unittest.main()
