import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import CondLogNormal as BaseCondLogNormal
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondLogNormal


class TestLogNormal(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        log_normal = CondLogNormal(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"mean": 0.0, "std": 1.0}

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        log_normal = CondLogNormal(Scope([0], [1]))

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.5], [1.0], [1.5]])
        targets = torch.tensor([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": mean, "std": std}
        )
        node_log_normal = BaseCondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": mean, "std": std}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_log_normal, data)
        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        mean = torch.tensor(random.random(), requires_grad=True)
        std = torch.tensor(
            random.random() + 1e-7, requires_grad=True
        )  # offset by small number to avoid zero

        torch_log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": mean, "std": std}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(mean.grad is not None)
        self.assertTrue(std.grad is not None)


"""
    def test_likelihood_marginalization(self):

        log_normal = LogNormal(Scope([0], [1]), 0.0, 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(log_normal, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Log-Normal distribution: floats (0,inf)

        log_normal = LogNormal(Scope([0], [1]), 0.0, 1.0)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, log_normal, torch.tensor([[float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, log_normal, torch.tensor([[-float("inf")]]))

        # invalid float values
        self.assertRaises(ValueError, log_likelihood, log_normal, torch.tensor([[0]]))

        # valid float values
        log_likelihood(
            log_normal, torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]])
        )
        log_likelihood(log_normal, torch.tensor([[4.3]]))
    """

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
