from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson as BaseCondPoisson,
)
from spflow.base.inference.nodes.leaves.parametric.cond_poisson import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.cond_poisson import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestPoisson(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": 1.0}

        poisson = CondPoisson(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        poisson = CondPoisson(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"l": 1.0}

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        poisson = CondPoisson(Scope([0], [1]))

        cond_f = lambda data: {"l": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[poisson] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [2], [5]])
        targets = torch.tensor([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        l = random.randint(1, 10)

        torch_poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})
        node_poisson = BaseCondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data)
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        l = torch.tensor(
            random.randint(1, 10),
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )

        torch_poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(l.grad is not None)

    def test_likelihood_marginalization(self):

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Poisson distribution: integers N U {0}

        l = random.random()

        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, poisson, torch.tensor([[-float("inf")]])
        )
        self.assertRaises(
            ValueError, log_likelihood, poisson, torch.tensor([[float("inf")]])
        )

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError, log_likelihood, poisson, torch.tensor([[-1]])
        )

        # check valid integers within valid range
        log_likelihood(poisson, torch.tensor([[0]]))
        log_likelihood(poisson, torch.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]
            ),
        )
        self.assertRaises(
            ValueError, log_likelihood, poisson, torch.tensor([[10.1]])
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
