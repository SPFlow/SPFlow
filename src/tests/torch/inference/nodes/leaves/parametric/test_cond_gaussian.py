from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian as BaseCondGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.cond_gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.cond_gaussian import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        gaussian = CondGaussian(Scope([0]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0], [1.0]])
        targets = torch.tensor([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        gaussian = CondGaussian(Scope([0]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gaussian] = {"mean": 0.0, "std": 1.0}

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0], [1.0]])
        targets = torch.tensor([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gaussian = CondGaussian(Scope([0]))

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gaussian] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0], [1.0]])
        targets = torch.tensor([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": mean, "std": std}
        )
        node_gaussian = BaseCondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": mean, "std": std}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs = log_likelihood(node_gaussian, data)
        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        mean = torch.tensor(random.random(), requires_grad=True)
        std = torch.tensor(
            random.random() + 1e-7, requires_grad=True
        )  # offset by small number to avoid zero

        torch_gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": mean, "std": std}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(mean.grad is not None)
        self.assertTrue(std.grad is not None)

    def test_likelihood_marginalization(self):

        gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(gaussian, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Gaussian distribution: floats (-inf, inf)

        gaussian = CondGaussian(
            Scope([0]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, gaussian, torch.tensor([[float("inf")]])
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            gaussian,
            torch.tensor([[-float("inf")]]),
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
