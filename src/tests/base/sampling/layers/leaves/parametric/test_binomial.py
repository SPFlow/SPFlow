from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.binomial import (
    BinomialLayer,
)
from spflow.base.inference.layers.leaves.parametric.binomial import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.binomial import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.base.inference.nodes.leaves.parametric.binomial import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.binomial import sample
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

        binomial_layer = BinomialLayer(
            scope=Scope([0]), n=3, p=[0.8, 0.3], n_nodes=2
        )
        s1 = SPNSumNode(children=[binomial_layer], weights=[0.3, 0.7])

        binomial_nodes = [
            Binomial(Scope([0]), n=3, p=0.8),
            Binomial(Scope([0]), n=3, p=0.3),
        ]
        s2 = SPNSumNode(children=binomial_nodes, weights=[0.3, 0.7])

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

        binomial_layer = BinomialLayer(
            scope=[Scope([0]), Scope([1])], n=[3, 5], p=[0.8, 0.3]
        )
        p1 = SPNProductNode(children=[binomial_layer])

        binomial_nodes = [
            Binomial(Scope([0]), n=3, p=0.8),
            Binomial(Scope([1]), n=5, p=0.3),
        ]
        p2 = SPNProductNode(children=binomial_nodes)

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

        binomial_layer = BinomialLayer(
            scope=Scope([0]), n=3, p=[0.8, 0.3], n_nodes=2
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, binomial_layer)


if __name__ == "__main__":
    unittest.main()
