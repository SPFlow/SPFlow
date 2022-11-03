from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.base.inference import log_likelihood, likelihood
from spflow.base.structure.spn import (
    SumNode,
    ProductNode,
    CondNegativeBinomial,
    CondNegativeBinomialLayer,
)
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_likelihood_no_p(self):

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2
        )
        self.assertRaises(
            ValueError, log_likelihood, negative_binomial, np.array([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [1.0, 1.0]}

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2, cond_f=cond_f
        )

        # create test inputs/outputs
        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_p(self):

        negative_binomial = CondNegativeBinomialLayer(
            Scope([0], [1]), n=2, n_nodes=2
        )

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"p": [1.0, 1.0]}

        # create test inputs/outputs
        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(
            negative_binomial, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)

        cond_f = lambda data: {"p": np.array([1.0, 1.0])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_layer_likelihood_1(self):

        negative_binomial_layer = CondNegativeBinomialLayer(
            scope=Scope([0], [1]),
            n=3,
            cond_f=lambda data: {"p": [0.8, 0.3]},
            n_nodes=2,
        )
        s1 = SumNode(children=[negative_binomial_layer], weights=[0.3, 0.7])

        negative_binomial_nodes = [
            CondNegativeBinomial(
                Scope([0], [1]), n=3, cond_f=lambda data: {"p": 0.8}
            ),
            CondNegativeBinomial(
                Scope([0], [1]), n=3, cond_f=lambda data: {"p": 0.3}
            ),
        ]
        s2 = SumNode(children=negative_binomial_nodes, weights=[0.3, 0.7])

        data = np.array([[0], [1], [0]])

        self.assertTrue(
            np.all(log_likelihood(s1, data) == log_likelihood(s2, data))
        )

    def test_layer_likelihood_2(self):

        negative_binomial_layer = CondNegativeBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            n=[3, 5],
            cond_f=lambda data: {"p": [0.8, 0.3]},
        )
        p1 = ProductNode(children=[negative_binomial_layer])

        negative_binomial_nodes = [
            CondNegativeBinomial(
                Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.8}
            ),
            CondNegativeBinomial(
                Scope([1], [2]), n=5, cond_f=lambda data: {"p": 0.3}
            ),
        ]
        p2 = ProductNode(children=negative_binomial_nodes)

        data = np.array([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(
            np.all(log_likelihood(p1, data) == log_likelihood(p2, data))
        )


if __name__ == "__main__":
    unittest.main()
