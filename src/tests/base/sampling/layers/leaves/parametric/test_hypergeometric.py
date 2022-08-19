from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer
from spflow.base.inference.layers.leaves.parametric.hypergeometric import log_likelihood
from spflow.base.sampling.layers.leaves.parametric.hypergeometric import sample
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.base.inference.nodes.leaves.parametric.hypergeometric import log_likelihood
from spflow.base.sampling.nodes.leaves.parametric.hypergeometric import sample
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        hypergeometric_layer = HypergeometricLayer(scope=Scope([0]), N=8, M=3, n=4, n_nodes=2)
        s1 = SPNSumNode(children=[hypergeometric_layer], weights=[0.3, 0.7])

        hypergeometric_nodes = [Hypergeometric(Scope([0]), N=8, M=3, n=4), Hypergeometric(Scope([0]), N=8, M=3, n=4)]
        s2 = SPNSumNode(children=hypergeometric_nodes, weights=[0.3, 0.7])

        layer_samples = sample(s1, 10000)
        nodes_samples = sample(s2, 10000)
        self.assertTrue(np.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))

    def test_sampling_2(self):

        hypergeometric_layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[8, 10], M=[3, 2], n=[4, 5])
        p1 = SPNProductNode(children=[hypergeometric_layer])

        hypergeometric_nodes = [Hypergeometric(Scope([0]), N=8, M=3, n=4), Hypergeometric(Scope([1]), N=10, M=2, n=5)]
        p2 = SPNProductNode(children=hypergeometric_nodes)

        layer_samples = sample(p1, 10000)
        nodes_samples = sample(p2, 10000)
        self.assertTrue(np.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))


if __name__ == "__main__":
    unittest.main()