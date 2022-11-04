from spflow.meta.data import Scope
from spflow.base.structure.spn import (
    SumNode,
    ProductNode,
    CondMultivariateGaussian,
    CondMultivariateGaussianLayer,
)
from spflow.base.inference import log_likelihood
from spflow.base.sampling import sample
import numpy as np
import random
import unittest


class TestNode(unittest.TestCase):
    def test_sampling_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=Scope([0, 1], [2]),
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [
                    [[0.13, 0.08], [0.08, 0.05]],
                    [[0.17, 0.054], [0.054, 0.0296]],
                ],
            },
            n_nodes=2,
        )
        s1 = SumNode(children=[multivariate_gaussian_layer], weights=[0.3, 0.7])

        multivariate_gaussian_nodes = [
            CondMultivariateGaussian(
                Scope([0, 1], [2]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[0.13, 0.08], [0.08, 0.05]],
                },
            ),
            CondMultivariateGaussian(
                Scope([0, 1], [2]),
                cond_f=lambda data: {
                    "mean": [0.2, -0.1],
                    "cov": [[0.17, 0.054], [0.054, 0.0296]],
                },
            ),
        ]
        s2 = SumNode(children=multivariate_gaussian_nodes, weights=[0.3, 0.7])

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

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=[Scope([0, 1], [4]), Scope([2, 3], [4])],
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [
                    [[0.13, 0.08], [0.08, 0.05]],
                    [[0.17, 0.054], [0.054, 0.0296]],
                ],
            },
        )
        p1 = ProductNode(children=[multivariate_gaussian_layer])

        multivariate_gaussian_nodes = [
            CondMultivariateGaussian(
                Scope([0, 1], [4]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[0.13, 0.08], [0.08, 0.05]],
                },
            ),
            CondMultivariateGaussian(
                Scope([2, 3], [4]),
                cond_f=lambda data: {
                    "mean": [0.2, -0.1],
                    "cov": [[0.17, 0.054], [0.054, 0.0296]],
                },
            ),
        ]
        p2 = ProductNode(children=multivariate_gaussian_nodes)

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

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=Scope([0, 1], [2]),
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [
                    [[0.13, 0.08], [0.08, 0.05]],
                    [[0.17, 0.054], [0.054, 0.0296]],
                ],
            },
            n_nodes=2,
        )

        # check if empty output ids (i.e., []) works AND sampling from non-disjoint scopes fails
        self.assertRaises(ValueError, sample, multivariate_gaussian_layer)


if __name__ == "__main__":
    unittest.main()
