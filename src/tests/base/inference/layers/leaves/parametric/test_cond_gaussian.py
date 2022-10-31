from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.cond_gaussian import (
    CondGaussianLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_gaussian import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.cond_gaussian import (
    log_likelihood,
)
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood, likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_mean(self):

        gaussian = CondGaussianLayer(
            Scope([0], [1]), cond_f=lambda data: {"std": [1.0, 1.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, gaussian, np.array([[0], [1]])
        )

    def test_likelihood_no_std(self):

        gaussian = CondGaussianLayer(
            Scope([0], [1]), cond_f=lambda data: {"mean": [0.0, 0.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, gaussian, np.array([[0], [1]])
        )

    def test_likelihood_no_mean_std(self):

        gaussian = CondGaussianLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, gaussian, np.array([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [1.0, 1.0]}

        gaussian = CondGaussianLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0.0], [1.0], [-1.0]])
        targets = np.array([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        gaussian = CondGaussianLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gaussian] = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}

        # create test inputs/outputs
        data = np.array([[0.0], [1.0], [-1.0]])
        targets = np.array([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gaussian = CondGaussianLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [1.0, 1.0]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gaussian] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0.0], [1.0], [-1.0]])
        targets = np.array([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        gaussian_layer = CondGaussianLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[gaussian_layer], weights=[0.3, 0.7])

        gaussian_nodes = [
            CondGaussian(
                Scope([0], [1]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondGaussian(
                Scope([0], [1]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        s2 = SPNSumNode(children=gaussian_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        gaussian_layer = CondGaussianLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
        )
        p1 = SPNProductNode(children=[gaussian_layer])

        gaussian_nodes = [
            CondGaussian(
                Scope([0], [2]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondGaussian(
                Scope([1], [2]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        p2 = SPNProductNode(children=gaussian_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
