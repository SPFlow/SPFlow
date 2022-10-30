from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial as BaseCondNegativeBinomial,
)
from spflow.base.inference.nodes.leaves.parametric.cond_negative_binomial import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.cond_negative_binomial import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestNegativeBinomial(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": 1.0}

        negative_binomial = CondNegativeBinomial(Scope([0]), n=1, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        negative_binomial = CondNegativeBinomial(Scope([0]), n=1)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"p": 1.0}

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(
            negative_binomial, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        negative_binomial = CondNegativeBinomial(Scope([0]), n=1)

        cond_f = lambda data: {"p": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(
            negative_binomial, data, dispatch_ctx=dispatch_ctx
        )

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = CondNegativeBinomial(
            Scope([0]), n, cond_f=lambda data: {"p": p}
        )
        node_negative_binomial = BaseCondNegativeBinomial(
            Scope([0]), n, cond_f=lambda data: {"p": p}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_negative_binomial, data)
        log_probs_torch = log_likelihood(
            torch_negative_binomial, torch.tensor(data)
        )

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        n = random.randint(2, 10)
        p = torch.tensor(random.random(), requires_grad=True)

        torch_negative_binomial = CondNegativeBinomial(
            Scope([0]), n, cond_f=lambda data: {"p": p}
        )

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs_torch = log_likelihood(
            torch_negative_binomial, torch.tensor(data)
        )

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_negative_binomial.n.grad is None)
        self.assertTrue(p.grad is not None)

    def test_likelihood_p_1(self):

        # p = 1
        negative_binomial = CondNegativeBinomial(
            Scope([0]), 1, cond_f=lambda data: {"p": 1.0}
        )

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_n_float(self):

        negative_binomial = CondNegativeBinomial(
            Scope([0]), 1, cond_f=lambda data: {"p": 0.5}
        )
        self.assertRaises(Exception, likelihood, negative_binomial, 0.5)

    def test_likelihood_marginalization(self):

        negative_binomial = CondNegativeBinomial(
            Scope([0]), 20, cond_f=lambda data: {"p": 0.3}
        )
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Negative Binomial distribution: integers N U {0}

        n = 20
        p = 0.3

        negative_binomial = CondNegativeBinomial(
            Scope([0]), n, cond_f=lambda data: {"p": p}
        )

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[-float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[float("inf")]]),
        )

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError, log_likelihood, negative_binomial, torch.tensor([[-1]])
        )

        # check valid integers within valid range
        log_likelihood(negative_binomial, torch.tensor([[0]]))
        log_likelihood(negative_binomial, torch.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[10.1]]),
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
