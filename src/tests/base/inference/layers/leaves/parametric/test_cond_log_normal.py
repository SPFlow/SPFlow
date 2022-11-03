from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.layers.leaves.parametric.cond_log_normal import (
    CondLogNormalLayer,
)
from spflow.base.inference.layers.leaves.parametric.cond_log_normal import (
    log_likelihood,
)
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from spflow.base.inference.nodes.leaves.parametric.cond_log_normal import (
    log_likelihood,
)
from spflow.base.structure.spn.nodes.sum_node import SPNSumNode
from spflow.base.inference.spn.nodes.sum_node import log_likelihood
from spflow.base.structure.spn.nodes.product_node import SPNProductNode
from spflow.base.inference.spn.nodes.product_node import log_likelihood
from spflow.base.inference.module import log_likelihood, likelihood
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_mean(self):

        log_normal = CondLogNormalLayer(
            Scope([0], [1]),
            cond_f=lambda data: {"std": [0.25, 0.25]},
            n_nodes=2,
        )
        self.assertRaises(
            KeyError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_no_std(self):

        log_normal = CondLogNormalLayer(
            Scope([0], [1]), cond_f=lambda data: {"mean": [0.0, 0.0]}, n_nodes=2
        )
        self.assertRaises(
            KeyError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_no_mean_std(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [0.25, 0.25]}

        log_normal = CondLogNormalLayer(
            Scope([0], [1]), n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {
            "mean": [0.0, 0.0],
            "std": [0.25, 0.25],
        }

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        log_normal = CondLogNormalLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"mean": [0.0, 0.0], "std": [0.25, 0.25]}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        log_normal_layer = CondLogNormalLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
            n_nodes=2,
        )
        s1 = SPNSumNode(children=[log_normal_layer], weights=[0.3, 0.7])

        log_normal_nodes = [
            CondLogNormal(
                Scope([0], [1]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondLogNormal(
                Scope([0], [1]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        s2 = SPNSumNode(children=log_normal_nodes, weights=[0.3, 0.7])

        data = np.array([[0.5], [1.5], [0.3]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        log_normal_layer = CondLogNormalLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"mean": [0.8, 0.3], "std": [1.3, 0.4]},
        )
        p1 = SPNProductNode(children=[log_normal_layer])

        log_normal_nodes = [
            CondLogNormal(
                Scope([0], [2]), cond_f=lambda data: {"mean": 0.8, "std": 1.3}
            ),
            CondLogNormal(
                Scope([1], [2]), cond_f=lambda data: {"mean": 0.3, "std": 0.4}
            ),
        ]
        p2 = SPNProductNode(children=log_normal_nodes)

        data = np.array([[0.5, 1.6], [0.1, 0.3], [0.47, 0.7]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
