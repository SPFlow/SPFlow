from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.gamma import GammaLayer
from spflow.base.inference.layers.leaves.parametric.gamma import log_likelihood
from spflow.base.sampling.layers.leaves.parametric.gamma import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.inference.nodes.leaves.parametric.gamma import log_likelihood
from spflow.base.sampling.nodes.leaves.parametric.gamma import sample
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
import random
import unittest


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        gamma_layer = GammaLayer(
            scope=Scope([0]), alpha=[0.8, 0.3], beta=[1.3, 0.4], n_nodes=2
        )
        s1 = SPNSumNode(children=[gamma_layer], weights=[0.3, 0.7])

        gamma_nodes = [
            Gamma(Scope([0]), alpha=0.8, beta=1.3),
            Gamma(Scope([0]), alpha=0.3, beta=0.4),
        ]
        s2 = SPNSumNode(children=gamma_nodes, weights=[0.3, 0.7])

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

        gamma_layer = GammaLayer(
            scope=[Scope([0]), Scope([1])], alpha=[0.8, 0.3], beta=[1.3, 0.4]
        )
        p1 = SPNProductNode(children=[gamma_layer])

        gamma_nodes = [
            Gamma(Scope([0]), alpha=0.8, beta=1.3),
            Gamma(Scope([1]), alpha=0.3, beta=0.4),
        ]
        p2 = SPNProductNode(children=gamma_nodes)

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

        gamma_layer = GammaLayer(
            scope=Scope([0]), alpha=[0.8, 0.3], beta=[1.3, 0.4], n_nodes=2
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, gamma_layer)


if __name__ == "__main__":
    unittest.main()
