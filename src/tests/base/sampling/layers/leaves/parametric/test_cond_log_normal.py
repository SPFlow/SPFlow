from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.cond_log_normal import (
    CondLogNormalLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_log_normal import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.cond_log_normal import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from spflow.base.inference.nodes.leaves.parametric.cond_log_normal import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.cond_log_normal import sample
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

        log_normal_layer = CondLogNormalLayer(
            scope=Scope([0]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[log_normal_layer], weights=[0.3, 0.7])

        log_normal_nodes = [
            CondLogNormal(
                Scope([0]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondLogNormal(
                Scope([0]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        s2 = SPNSumNode(children=log_normal_nodes, weights=[0.3, 0.7])

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

        log_normal_layer = CondLogNormalLayer(
            scope=[Scope([0]), Scope([1])],
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
        )
        p1 = SPNProductNode(children=[log_normal_layer])

        log_normal_nodes = [
            CondLogNormal(
                Scope([0]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondLogNormal(
                Scope([1]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        p2 = SPNProductNode(children=log_normal_nodes)

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

        log_normal_layer = CondLogNormalLayer(
            scope=Scope([0]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, log_normal_layer)


if __name__ == "__main__":
    unittest.main()
