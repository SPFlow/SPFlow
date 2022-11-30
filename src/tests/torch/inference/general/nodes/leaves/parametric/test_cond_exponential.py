import random
import unittest

import numpy as np
import torch
from packaging import version

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import CondExponential as BaseCondExponential
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondExponential


class TestExponential(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": 0.5}

        exponential = CondExponential(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394], [0.0410425]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        exponential = CondExponential(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"l": 0.5}

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394], [0.0410425]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        exponential = CondExponential(Scope([0], [1]))

        cond_f = lambda data: {"l": 0.5}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[2], [5]])
        targets = torch.tensor([[0.18394], [0.0410425]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        l = random.random() + 1e-7  # small offset to avoid zero

        torch_exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})
        node_exponential = BaseCondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_exponential, data)
        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        l = torch.tensor(random.random(), requires_grad=True)

        torch_exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(l.grad is not None)

    def test_likelihood_marginalization(self):

        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Exponential distribution: floats [0,inf) (note: 0 excluded in pytorch support)

        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.5})

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            torch.tensor([[-float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            torch.tensor([[float("inf")]]),
        )

        # check valid float values (within range)
        log_likelihood(
            exponential,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        log_likelihood(exponential, torch.tensor([[10.5]]))

        # check invalid float values (outside range)
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )

        if version.parse(torch.__version__) < version.parse("1.11.0"):
            # edge case 0 (part of the support in scipy, but NOT pytorch)
            self.assertRaises(ValueError, log_likelihood, exponential, torch.tensor([[0.0]]))
        else:
            # edge case 0
            log_likelihood(exponential, torch.tensor([[0.0]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
