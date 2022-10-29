from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.bernoulli import (
    BernoulliLayer,
)
from spflow.base.inference.layers.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.bernoulli import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.base.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.bernoulli import sample
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

        bernoulli_layer = BernoulliLayer(
            scope=Scope([0]), p=[0.8, 0.3], n_nodes=2
        )
        s1 = SPNSumNode(children=[bernoulli_layer], weights=[0.3, 0.7])

        bernoulli_nodes = [
            Bernoulli(Scope([0]), p=0.8),
            Bernoulli(Scope([0]), p=0.3),
        ]
        s2 = SPNSumNode(children=bernoulli_nodes, weights=[0.3, 0.7])

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

        bernoulli_layer = BernoulliLayer(
            scope=[Scope([0]), Scope([1])], p=[0.8, 0.3]
        )
        p1 = SPNProductNode(children=[bernoulli_layer])

        bernoulli_nodes = [
            Bernoulli(Scope([0]), p=0.8),
            Bernoulli(Scope([1]), p=0.3),
        ]
        p2 = SPNProductNode(children=bernoulli_nodes)

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

        bernoulli_layer = BernoulliLayer(
            scope=[Scope([0]), Scope([0])], p=[0.8, 0.3]
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, bernoulli_layer)


if __name__ == "__main__":
    unittest.main()
