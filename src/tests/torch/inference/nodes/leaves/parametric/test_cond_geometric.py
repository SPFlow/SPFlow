from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_geometric import (
    CondGeometric as BaseCondGeometric,
)
from spflow.base.inference.nodes.leaves.parametric.cond_geometric import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_geometric import (
    CondGeometric,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.cond_geometric import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestGeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        p = random.random()
        cond_f = lambda data: {"p": 0.5}

        geometric = CondGeometric(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        geometric = CondGeometric(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"p": 0.5}

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        geometric = CondGeometric(Scope([0], [1]))

        cond_f = lambda data: {"p": 0.5}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[1], [5], [10]])
        targets = torch.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        p = random.random()

        torch_geometric = CondGeometric(
            Scope([0], [1]), cond_f=lambda data: {"p": p}
        )
        node_geometric = BaseCondGeometric(
            Scope([0], [1]), cond_f=lambda data: {"p": p}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs = log_likelihood(node_geometric, data)
        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        p = torch.tensor(random.random(), requires_grad=True)

        torch_geometric = CondGeometric(
            Scope([0], [1]), cond_f=lambda data: {"p": p}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)

    def test_likelihood_marginalization(self):

        geometric = CondGeometric(
            Scope([0], [1]), cond_f=lambda data: {"p": 0.5}
        )
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(geometric, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Geometric distribution: integers N\{0}

        geometric = CondGeometric(
            Scope([0], [1]), cond_f=lambda data: {"p": 0.5}
        )

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            torch.tensor([[float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            torch.tensor([[-float("inf")]]),
        )

        # valid integers, but outside valid range
        self.assertRaises(
            ValueError, log_likelihood, geometric, torch.tensor([[0.0]])
        )

        # valid integers within valid range
        data = torch.tensor([[1], [10]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # invalid floats
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            torch.tensor(
                [[torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            torch.tensor(
                [[torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]
            ),
        )
        self.assertRaises(
            ValueError, log_likelihood, geometric, torch.tensor([[1.5]])
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
