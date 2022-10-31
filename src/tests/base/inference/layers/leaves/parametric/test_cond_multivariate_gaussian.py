from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussianLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_multivariate_gaussian import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.cond_multivariate_gaussian import (
    log_likelihood,
)
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood, likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_mean(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]),
            cond_f=lambda data: {
                "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
            },
            n_nodes=2,
        )
        self.assertRaises(
            KeyError,
            log_likelihood,
            multivariate_gaussian,
            np.array([[0], [1]]),
        )

    def test_likelihood_no_cov(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"mean": [[0.0, 0.0], [0.0, 0.0]]},
            n_nodes=2,
        )
        self.assertRaises(
            KeyError,
            log_likelihood,
            multivariate_gaussian,
            np.array([[0], [1]]),
        )

    def test_likelihood_no_mean_cov(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0], [1]), n_nodes=2
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            np.array([[0], [1]]),
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)
        targets = np.array([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]), n_nodes=2
        )

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)
        targets = np.array([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(
            multivariate_gaussian, data, dispatch_ctx=dispatch_ctx
        )
        log_probs = log_likelihood(
            multivariate_gaussian, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]), n_nodes=2
        )

        cond_f = lambda data: {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)
        targets = np.array([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(
            multivariate_gaussian, data, dispatch_ctx=dispatch_ctx
        )
        log_probs = log_likelihood(
            multivariate_gaussian, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=Scope([0, 1], [2]),
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]],
            },
            n_nodes=2,
        )
        s1 = SPNSumNode(
            children=[multivariate_gaussian_layer], weights=[0.3, 0.7]
        )

        multivariate_gaussian_nodes = [
            CondMultivariateGaussian(
                Scope([0, 1], [2]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[1.3, 0.4], [0.4, 0.5]],
                },
            ),
            CondMultivariateGaussian(
                Scope([0, 1], [2]),
                cond_f=lambda data: {
                    "mean": [0.2, -0.1],
                    "cov": [[0.5, 0.1], [0.1, 1.4]],
                },
            ),
        ]
        s2 = SPNSumNode(
            children=multivariate_gaussian_nodes, weights=[0.3, 0.7]
        )

        data = np.array([[0.5, 0.3], [1.5, -0.3], [0.3, 0.0]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=[Scope([0, 1], [4]), Scope([2, 3], [4])],
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]],
            },
        )
        p1 = SPNProductNode(children=[multivariate_gaussian_layer])

        multivariate_gaussian_nodes = [
            CondMultivariateGaussian(
                Scope([0, 1], [4]),
                cond_f=lambda data: {
                    "mean": [0.8, 0.3],
                    "cov": [[1.3, 0.4], [0.4, 0.5]],
                },
            ),
            CondMultivariateGaussian(
                Scope([2, 3], [4]),
                cond_f=lambda data: {
                    "mean": [0.2, -0.1],
                    "cov": [[0.5, 0.1], [0.1, 1.4]],
                },
            ),
        ]
        p2 = SPNProductNode(children=multivariate_gaussian_nodes)

        data = np.array(
            [
                [0.5, 1.6, -0.1, 3.0],
                [0.1, 0.3, 0.9, 0.73],
                [0.47, 0.7, 0.5, 0.1],
            ]
        )

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
