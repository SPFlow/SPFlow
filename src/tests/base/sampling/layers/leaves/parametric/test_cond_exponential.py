from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.cond_exponential import (
    CondExponentialLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_exponential import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.cond_exponential import (
    sample,
)
from spflow.base.structure.spn.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.spn.nodes.node import log_likelihood
from spflow.base.sampling.spn.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
)
from spflow.base.inference.nodes.leaves.parametric.cond_exponential import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.cond_exponential import sample
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

        exponential_layer = CondExponentialLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.8, 0.3]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[exponential_layer], weights=[0.3, 0.7])

        exponential_nodes = [
            CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 0.8}),
            CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 0.3}),
        ]
        s2 = SPNSumNode(children=exponential_nodes, weights=[0.3, 0.7])

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

        exponential_layer = CondExponentialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": [0.8, 0.3]},
        )
        p1 = SPNProductNode(children=[exponential_layer])

        exponential_nodes = [
            CondExponential(Scope([0], [2]), cond_f=lambda data: {"l": 0.8}),
            CondExponential(Scope([1], [2]), cond_f=lambda data: {"l": 0.3}),
        ]
        p2 = SPNProductNode(children=exponential_nodes)

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

        exponential_layer = CondExponentialLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.8, 0.3]},
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, exponential_layer)


if __name__ == "__main__":
    unittest.main()
