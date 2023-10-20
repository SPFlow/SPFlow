import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import (
    CondBinomial,
    CondBinomialLayer,
    ProductNode,
    SumNode,
)
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestNode(unittest.TestCase):
    def test_likelihood_no_p(self):

        binomial = CondBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [0.8, 0.5]}

        binomial = CondBinomialLayer(Scope([0], [1]), n=1, n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_p(self):

        binomial = CondBinomialLayer(Scope([0], [1]), n=1, n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {"p": [0.8, 0.5]}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondBinomialLayer(Scope([0], [1]), n=1, n_nodes=2)

        cond_f = lambda data: {"p": np.array([0.8, 0.5])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        binomial_layer = CondBinomialLayer(
            scope=[Scope([0], [1]), Scope([0], [1])],
            n=3,
            cond_f=lambda data: {"p": [0.8, 0.3]},
        )
        s1 = SumNode(children=[binomial_layer], weights=[0.3, 0.7])

        binomial_nodes = [
            CondBinomial(Scope([0], [1]), n=3, cond_f=lambda data: {"p": 0.8}),
            CondBinomial(Scope([0], [1]), n=3, cond_f=lambda data: {"p": 0.3}),
        ]
        s2 = SumNode(children=binomial_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [1], [0]])

        self.assertTrue(np.all(log_likelihood(s1, data) == log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        binomial_layer = CondBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            n=[3, 5],
            cond_f=lambda data: {"p": [0.8, 0.3]},
        )
        p1 = ProductNode(children=[binomial_layer])

        binomial_nodes = [
            CondBinomial(Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.8}),
            CondBinomial(Scope([1], [2]), n=5, cond_f=lambda data: {"p": 0.3}),
        ]
        p2 = ProductNode(children=binomial_nodes)

        data = np.array([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(np.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()