from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.cond_binomial import CondBinomialLayer
from spflow.torch.inference.layers.leaves.parametric.cond_binomial import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.cond_binomial import CondBinomial
from spflow.torch.inference.nodes.leaves.parametric.cond_binomial import log_likelihood
from spflow.torch.inference.module import log_likelihood, likelihood
import torch
import numpy as np
import unittest
import itertools
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_p(self):

        binomial = CondBinomialLayer(Scope([0]), n=2, n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {'p': [0.8, 0.5]}

        binomial = CondBinomialLayer(Scope([0]), n=1, n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        binomial = CondBinomialLayer(Scope([0]), n=1, n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {'p': [0.8, 0.5]}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_args_cond_f(self):

        bernoulli = CondBinomialLayer(Scope([0]), n=1, n_nodes=2)

        cond_f = lambda data: {'p': torch.tensor([0.8, 0.5])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {'cond_f': cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[0.2, 0.5], [0.8, 0.5]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondBinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], n=[3, 2, 3], cond_f=lambda data: {'p': [0.2, 0.5, 0.9]})

        nodes = [
            CondBinomial(Scope([0]), n=3, cond_f=lambda data: {'p': 0.2}),
            CondBinomial(Scope([1]), n=2, cond_f=lambda data: {'p': 0.5}),
            CondBinomial(Scope([0]), n=3, cond_f=lambda data: {'p': 0.9}),
        ]

        dummy_data = torch.tensor([[3, 1], [1, 2], [0, 0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        n = [4, 6]
        p = torch.tensor([random.random(), random.random()], requires_grad=True)

        torch_binomial = CondBinomialLayer(scope=[Scope([0]), Scope([1])], n=n, cond_f=lambda data: {'p': p})

        # create dummy input data (batch size x random variables)
        data = torch.tensor([[0, 5], [3, 2], [4, 1]])

        log_probs_torch = log_likelihood(torch_binomial, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_binomial.n.grad is None)
        self.assertTrue(p.grad is not None)

    def test_likelihood_marginalization(self):

        binomial = CondBinomialLayer(scope=[Scope([0]), Scope([1])], n=5, cond_f=lambda data: {'p': random.random()})
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(binomial, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))
    
    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()