import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_stack

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import (
    ProductNode,
    SumNode,
)
from spflow.tensorly.structure.general.nodes.leaves import CondMultivariateGaussian
from spflow.tensorly.structure.general.layers.leaves import CondMultivariateGaussianLayer
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestNode(unittest.TestCase):
    def test_likelihood_no_mean(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]},
            n_nodes=2,
        )
        self.assertRaises(
            KeyError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0], [1]]),
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
            tl.tensor([[0], [1]]),
        )

    def test_likelihood_no_mean_cov(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0], [1]]),
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2)

        cond_f = lambda data: {
            "mean": [[0.0, 0.0], [0.0, 0.0]],
            "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        }

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

        probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_layer_likelihood_1(self):

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=Scope([0, 1], [2]),
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]],
            },
            n_nodes=2,
        )
        s1 = SumNode(children=[multivariate_gaussian_layer], weights=[0.3, 0.7])

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
        s2 = SumNode(children=multivariate_gaussian_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0.5, 0.3], [1.5, -0.3], [0.3, 0.0]])

        self.assertTrue(tl_allclose(log_likelihood(s1, data), log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        multivariate_gaussian_layer = CondMultivariateGaussianLayer(
            scope=[Scope([0, 1], [4]), Scope([2, 3], [4])],
            cond_f=lambda data: {
                "mean": [[0.8, 0.3], [0.2, -0.1]],
                "cov": [[[1.3, 0.4], [0.4, 0.5]], [[0.5, 0.1], [0.1, 1.4]]],
            },
        )
        p1 = ProductNode(children=[multivariate_gaussian_layer])

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
        p2 = ProductNode(children=multivariate_gaussian_nodes)

        data = tl.tensor(
            [
                [0.5, 1.6, -0.1, 3.0],
                [0.1, 0.3, 0.9, 0.73],
                [0.47, 0.7, 0.5, 0.1],
            ]
        )

        self.assertTrue(tl_allclose(log_likelihood(p1, data), log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
