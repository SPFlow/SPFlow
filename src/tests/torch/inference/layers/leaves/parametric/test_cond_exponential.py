from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.layers.leaves.parametric.cond_exponential import CondExponentialLayer
from spflow.torch.inference.layers.leaves.parametric.cond_exponential import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.cond_exponential import CondExponential
from spflow.torch.inference.nodes.leaves.parametric.cond_exponential import log_likelihood
from spflow.torch.inference.module import log_likelihood, likelihood
import torch
import unittest
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_no_l(self):

        exponential = CondExponentialLayer(Scope([0]), n_nodes=2)
        self.assertRaises(ValueError, log_likelihood, exponential, torch.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {'l': [0.5, 1.0]}

        exponential = CondExponentialLayer(Scope([0]), n_nodes=2, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_args_l(self):

        exponential = CondExponentialLayer(Scope([0]), n_nodes=2)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {'l': [0.5, 1.0]}

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        exponential = CondExponentialLayer(Scope([0]), n_nodes=2)

        cond_f = lambda data: {'l': torch.tensor([0.5, 1.0])}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {'cond_f': cond_f}

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_layer_likelihood(self):

        layer = CondExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], cond_f=lambda data: {'l': [0.2, 1.0, 2.3]})

        nodes = [
            CondExponential(Scope([0]), cond_f=lambda data: {'l': 0.2}),
            CondExponential(Scope([1]), cond_f=lambda data: {'l': 1.0}),
            CondExponential(Scope([0]), cond_f=lambda data: {'l': 2.3}),
        ]

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

        l = torch.tensor([random.random(), random.random()], requires_grad=True)

        torch_exponential = CondExponentialLayer(scope=[Scope([0]), Scope([1])], cond_f=lambda data: {'l': l})

        # create dummy input data (batch size x random variables)
        data = torch.rand(3, 2)

        log_probs_torch = log_likelihood(torch_exponential, data)

        # create dummy targets
        targets_torch = torch.ones(3, 2)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(l.grad is not None)

    def test_likelihood_marginalization(self):
        
        exponential = CondExponentialLayer(scope=[Scope([0]), Scope([1])], cond_f=lambda data: {'l': random.random()+1e-7})
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(exponential, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()