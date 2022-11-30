import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import (
    CondExponential,
    CondExponentialLayer,
    ProductNode,
    SumNode,
)
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestNode(unittest.TestCase):
    def test_likelihood_no_l(self):

        exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, exponential, np.array([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": [0.5, 1.0]}

        exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5, 1.0], [0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_l(self):

        exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"l": [0.5, 1.0]}

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5, 1.0], [0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"l": np.array([0.5, 1.0])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5, 1.0], [0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        exponential_layer = CondExponentialLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"l": [0.5, 1.0]},
            n_nodes=2,
        )
        s1 = SumNode(children=[exponential_layer], weights=[0.3, 0.7])

        exponential_nodes = [
            CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 0.5}),
            CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.0}),
        ]
        s2 = SumNode(children=exponential_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [2], [5]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        exponential_layer = CondExponentialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"l": [0.5, 1.0]},
        )
        p1 = ProductNode(children=[exponential_layer])

        exponential_nodes = [
            CondExponential(Scope([0], [2]), cond_f=lambda data: {"l": 0.5}),
            CondExponential(Scope([1], [2]), cond_f=lambda data: {"l": 1.0}),
        ]
        p2 = ProductNode(children=exponential_nodes)

        data = np.array([[0, 0], [2, 2], [5, 5]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
