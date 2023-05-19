import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import (
    ProductNode,
    SumNode,
)
from spflow.tensorly.structure.general.nodes.leaves import CondBernoulli
from spflow.tensorly.structure.general.layers.leaves import CondBernoulliLayer
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestNode(unittest.TestCase):
    def test_likelihood_no_p(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": [0.8, 0.5]}

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_p(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"p": [0.8, 0.5]}

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

        cond_f = lambda data: {"p": tl.tensor([0.8, 0.5])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_layer_likelihood_1(self):

        bernoulli_layer = CondBernoulliLayer(
            scope=Scope([0], [1]),
            cond_f=lambda data: {"p": [0.8, 0.3]},
            n_nodes=2,
        )
        s1 = SumNode(children=[bernoulli_layer], weights=[0.3, 0.7])

        bernoulli_nodes = [
            CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.8}),
            CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.3}),
        ]
        s2 = SumNode(children=bernoulli_nodes, weights=[0.3, 0.7])

        data = tl.tensor([[0], [1], [0]])

        self.assertTrue(tl_allclose(log_likelihood(s1, data), log_likelihood(s2, data)))

    def test_layer_likelihood_2(self):

        bernoulli_layer = CondBernoulliLayer(
            scope=[Scope([0], [2]), Scope([1], [2])],
            cond_f=lambda data: {"p": [0.8, 0.3]},
        )
        p1 = ProductNode(children=[bernoulli_layer])

        bernoulli_nodes = [
            CondBernoulli(Scope([0], [2]), cond_f=lambda data: {"p": 0.8}),
            CondBernoulli(Scope([1], [2]), cond_f=lambda data: {"p": 0.3}),
        ]
        p2 = ProductNode(children=bernoulli_nodes)

        data = tl.tensor([[0, 1], [1, 1], [0, 0]])

        self.assertTrue(tl.all(log_likelihood(p1, data) == log_likelihood(p2, data)))


if __name__ == "__main__":
    unittest.main()
