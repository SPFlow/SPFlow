import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import (
    ProductNode,
    SumNode,
)
from spflow.tensorly.structure.general.nodes.leaves import CondLogNormal
from spflow.tensorly.structure.general.layers.leaves import CondLogNormalLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        log_normal_layer = CondLogNormalLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SumNode(children=[log_normal_layer], weights=[0.3, 0.7])

        log_normal_nodes = [
            CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}),
            CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}),
        ]
        s2 = SumNode(children=log_normal_nodes, weights=[0.3, 0.7])

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

        log_normal_layer = CondLogNormalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
        )
        p1 = ProductNode(children=[log_normal_layer])

        log_normal_nodes = [
            CondLogNormal(Scope([0], [2]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}),
            CondLogNormal(Scope([1], [2]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}),
        ]
        p2 = ProductNode(children=log_normal_nodes)

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

        log_normal_layer = CondLogNormalLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, log_normal_layer)


if __name__ == "__main__":
    unittest.main()
