from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.cond_poisson import (
    CondPoissonLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_poisson import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.cond_poisson import sample
from spflow.base.structure.spn.nodes.sum_node import SPNSumNode
from spflow.base.inference.spn.nodes.sum_node import log_likelihood
from spflow.base.sampling.spn.nodes.sum_node import sample
from spflow.base.structure.spn.nodes.product_node import SPNProductNode
from spflow.base.inference.spn.nodes.product_node import log_likelihood
from spflow.base.sampling.spn.nodes.product_node import sample
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
)
from spflow.base.inference.nodes.leaves.parametric.cond_poisson import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.cond_poisson import sample
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

        poisson_layer = CondPoissonLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.8, 0.3]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[poisson_layer], weights=[0.3, 0.7])

        poisson_nodes = [
            CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 0.8}),
            CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 0.3}),
        ]
        s2 = SPNSumNode(children=poisson_nodes, weights=[0.3, 0.7])

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

        poisson_layer = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": [0.8, 0.3]},
        )
        p1 = SPNProductNode(children=[poisson_layer])

        poisson_nodes = [
            CondPoisson(Scope([0], [2]), cond_f=lambda data: {"l": 0.8}),
            CondPoisson(Scope([1], [2]), cond_f=lambda data: {"l": 0.3}),
        ]
        p2 = SPNProductNode(children=poisson_nodes)

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

        poisson_layer = CondPoissonLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.8, 0.3]},
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, poisson_layer)


if __name__ == "__main__":
    unittest.main()
