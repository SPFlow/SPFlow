from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.cond_geometric import CondGeometricLayer
from spflow.base.inference.layers.leaves.parametric.cond_geometric import log_likelihood
from spflow.base.sampling.layers.leaves.parametric.cond_geometric import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.cond_geometric import CondGeometric
from spflow.base.inference.nodes.leaves.parametric.cond_geometric import log_likelihood
from spflow.base.sampling.nodes.leaves.parametric.cond_geometric import sample
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

        geometric_layer = CondGeometricLayer(scope=Scope([0]), cond_f=lambda data: {'p': [0.8, 0.3]}, n_nodes=2)
        s1 = SPNSumNode(children=[geometric_layer], weights=[0.3, 0.7])

        geometric_nodes = [CondGeometric(Scope([0]), cond_f=lambda data: {'p': 0.8}), CondGeometric(Scope([0]), cond_f=lambda data: {'p': 0.3})]
        s2 = SPNSumNode(children=geometric_nodes, weights=[0.3, 0.7])

        layer_samples = sample(s1, 10000)
        nodes_samples = sample(s2, 10000)
        self.assertTrue(np.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))

    def test_sampling_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        geometric_layer = CondGeometricLayer(scope=[Scope([0]), Scope([1])], cond_f=lambda data: {'p': [0.8, 0.3]})
        p1 = SPNProductNode(children=[geometric_layer])

        geometric_nodes = [CondGeometric(Scope([0]), cond_f=lambda data: {'p': 0.8}), CondGeometric(Scope([1]), cond_f=lambda data: {'p': 0.3})]
        p2 = SPNProductNode(children=geometric_nodes)

        layer_samples = sample(p1, 10000)
        nodes_samples = sample(p2, 10000)
        self.assertTrue(np.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))

    def test_sampling_3(self):
        
        geometric_layer = CondGeometricLayer(scope=Scope([0]), cond_f=lambda data: {'p': [0.8, 0.3]}, n_nodes=2)

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, geometric_layer)


if __name__ == "__main__":
    unittest.main()