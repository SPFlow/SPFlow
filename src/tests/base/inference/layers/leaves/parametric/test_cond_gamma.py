from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.cond_gamma import (
    CondGammaLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_gamma import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma
from spflow.base.inference.nodes.leaves.parametric.cond_gamma import (
    log_likelihood,
)
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.module import log_likelihood, likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_alpha(self):

        gamma = CondGammaLayer(
            Scope([0]), cond_f=lambda data: {"beta": [1.0, 1.0]}, n_nodes=2
        )
        self.assertRaises(KeyError, log_likelihood, gamma, np.array([[0], [1]]))

    def test_likelihood_no_beta(self):

        gamma = CondGammaLayer(
            Scope([0]), cond_f=lambda data: {"alpha": [1.0, 1.0]}, n_nodes=2
        )
        self.assertRaises(KeyError, log_likelihood, gamma, np.array([[0], [1]]))

    def test_likelihood_no_alpha_beta(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, gamma, np.array([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        gamma = CondGammaLayer(Scope([0]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = np.array(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        # create test inputs/outputs
        data = np.array([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = np.array(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gamma = CondGammaLayer(Scope([0]), n_nodes=2)

        cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
        targets = np.array(
            [[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]]
        )

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        gamma_layer = CondGammaLayer(
            scope=Scope([0]),
            cond_f=lambda data: {"alpha": [0.8, 0.3], "beta": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[gamma_layer], weights=[0.3, 0.7])

        gamma_nodes = [
            CondGamma(
                Scope([0]), cond_f=lambda data: {"alpha": 0.8, "beta": 1.3}
            ),
            CondGamma(
                Scope([0]), cond_f=lambda data: {"alpha": 0.3, "beta": 0.4}
            ),
        ]
        s2 = SPNSumNode(children=gamma_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        gamma_layer = CondGammaLayer(
            scope=[Scope([0]), Scope([1])],
            cond_f=lambda data: {"alpha": [0.8, 0.3], "beta": [1.3, 0.4]},
        )
        p1 = SPNProductNode(children=[gamma_layer])

        gamma_nodes = [
            CondGamma(
                Scope([0]), cond_f=lambda data: {"alpha": 0.8, "beta": 1.3}
            ),
            CondGamma(
                Scope([1]), cond_f=lambda data: {"alpha": 0.3, "beta": 0.4}
            ),
        ]
        p2 = SPNProductNode(children=gamma_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
