from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussianLayer,
)
from spflow.base.inference.layers.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.multivariate_gaussian import (
    sample,
)
from spflow.base.structure.spn.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.inference.spn.nodes.node import log_likelihood
from spflow.base.sampling.spn.nodes.node import sample
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.multivariate_gaussian import (
    sample,
)
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

        multivariate_gaussian_layer = MultivariateGaussianLayer(
            scope=Scope([0, 1]),
            mean=[[0.8, 0.3], [0.2, -0.1]],
            cov=[
                [[0.13, 0.08], [0.08, 0.05]],
                [[0.17, 0.054], [0.054, 0.0296]],
            ],
            n_nodes=2,
        )
        s1 = SPNSumNode(
            children=[multivariate_gaussian_layer], weights=[0.3, 0.7]
        )

        multivariate_gaussian_nodes = [
            MultivariateGaussian(
                Scope([0, 1]), mean=[0.8, 0.3], cov=[[0.13, 0.08], [0.08, 0.05]]
            ),
            MultivariateGaussian(
                Scope([0, 1]),
                mean=[0.2, -0.1],
                cov=[[0.17, 0.054], [0.054, 0.0296]],
            ),
        ]
        s2 = SPNSumNode(
            children=multivariate_gaussian_nodes, weights=[0.3, 0.7]
        )

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

        multivariate_gaussian_layer = MultivariateGaussianLayer(
            scope=[Scope([0, 1]), Scope([2, 3])],
            mean=[[0.8, 0.3], [0.2, -0.1]],
            cov=[
                [[0.13, 0.08], [0.08, 0.05]],
                [[0.17, 0.054], [0.054, 0.0296]],
            ],
        )
        p1 = SPNProductNode(children=[multivariate_gaussian_layer])

        multivariate_gaussian_nodes = [
            MultivariateGaussian(
                Scope([0, 1]), mean=[0.8, 0.3], cov=[[0.13, 0.08], [0.08, 0.05]]
            ),
            MultivariateGaussian(
                Scope([2, 3]),
                mean=[0.2, -0.1],
                cov=[[0.17, 0.054], [0.054, 0.0296]],
            ),
        ]
        p2 = SPNProductNode(children=multivariate_gaussian_nodes)

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

        multivariate_gaussian_layer = MultivariateGaussianLayer(
            scope=Scope([0, 1]),
            mean=[[0.8, 0.3], [0.2, -0.1]],
            cov=[
                [[0.13, 0.08], [0.08, 0.05]],
                [[0.17, 0.054], [0.054, 0.0296]],
            ],
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, multivariate_gaussian_layer)


if __name__ == "__main__":
    unittest.main()
