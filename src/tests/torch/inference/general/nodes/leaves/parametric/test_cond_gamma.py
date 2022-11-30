import random
import unittest

import numpy as np
import torch
from packaging import version

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import CondGamma as BaseCondGamma
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import CondGamma


class TestGamma(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

        gamma = CondGamma(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0.1], [1.0], [3.0]])
        targets = torch.tensor([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        gamma = CondGamma(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"alpha": 1.0, "beta": 1.0}

        # create test inputs/outputs
        data = torch.tensor([[0.1], [1.0], [3.0]])
        targets = torch.tensor([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gamma = CondGamma(Scope([0], [1]))

        cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.1], [1.0], [3.0]])
        targets = torch.tensor([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        alpha = torch.tensor(random.randint(1, 5), dtype=torch.get_default_dtype())
        beta = torch.tensor(random.randint(1, 5), dtype=torch.get_default_dtype())

        torch_gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})
        node_gamma = BaseCondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_gamma, data)
        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        alpha = torch.tensor(
            random.randint(1, 5),
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )
        beta = torch.tensor(
            random.randint(1, 5),
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )

        torch_gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(alpha.grad is not None)
        self.assertTrue(beta.grad is not None)

    def test_marginalization(self):

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(gamma, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Gamma distribution: floats (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})

        # TODO: 0

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, gamma, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, gamma, torch.tensor([[float("inf")]]))

        # check finite values > 0
        log_likelihood(
            gamma,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        log_likelihood(gamma, torch.tensor([[10.5]]))

        data = torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(all(data != 0.0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # check invalid float values (outside range)
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            # edge case 0
            self.assertRaises(ValueError, log_likelihood, gamma, torch.tensor([[0.0]]))
        else:
            # edge case 0
            log_likelihood(gamma, torch.tensor([[0.0]]))

        self.assertRaises(
            ValueError,
            log_likelihood,
            gamma,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
