import random
import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.sampling import sample
from spflow.base.structure.spn import CondCategorical, CondCategoricalLayer, ProductNode, SumNode
from spflow.meta.data import Scope


class TestCondCategorical(unittest.TestCase):
    def test_sampling_1(self):

        np.random.seed(0)
        random.seed(0)

        layer = CondCategoricalLayer(
            scope=Scope([0], [1]), 
            cond_f=lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}, 
            n_nodes=2
        )

        s1 = SumNode(children=[layer], weights=[0.3, 0.7])

        categorical_nodes = [
            CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]}), 
            CondCategorical(Scope([0], [1]), cond_f=lambda data: {"k": 2, "p": [0.3, 0.7]})
        ]
        s2 = SumNode(children=categorical_nodes, weights=[0.3, 0.7])

        layer_samples = sample(s1, 10000)
        node_samples = sample(s2, 10000)

        self.assertTrue(np.allclose(layer_samples.mean(axis=0), node_samples.mean(axis=0), atol=0.01, rtol=0.1))


    def test_sampling_2(self):

        np.random.seed(0)
        random.seed(0)

        layer = CondCategoricalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])], 
            cond_f= lambda data: {"k": [2, 2], "p": [[0.5, 0.5], [0.3, 0.7]]}
        )
        p1 = ProductNode(children=[layer])

        categorical_nodes = [
            CondCategorical(Scope([0], [2]), cond_f=lambda data: {"k": 2, "p": [0.5, 0.5]}), 
            CondCategorical(Scope([1], [2]), cond_f=lambda data: {"k": 2, "p": [0.3, 0.7]})
        ]
        p2 = ProductNode(children=categorical_nodes)

        layer_samples = sample(p1, 10000)
        node_samples = sample(p2, 10000)

        self.assertTrue(np.allclose(layer_samples.mean(axis=0), node_samples.mean(axis=0), atol=0.01, rtol=0.1))


if __name__ == "__main__":
    unittest.main()