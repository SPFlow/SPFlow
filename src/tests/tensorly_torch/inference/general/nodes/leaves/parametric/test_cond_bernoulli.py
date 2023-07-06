import random
import unittest

import numpy as np
import torch

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import CondBernoulli as BaseCondBernoulli
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
#from spflow.torch.structure.spn import CondBernoulli
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_bernoulli import CondBernoulli


class TestBernoulli(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        p = random.random()
        cond_f = lambda data: {"p": p}

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_p(self):

        bernoulli = CondBernoulli(Scope([0], [1]))

        p = random.random()
        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"p": p}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondBernoulli(Scope([0], [1]))

        p = random.random()
        cond_f = lambda data: {"p": p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        p = np.array(0.5)

        torch_bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": torch.tensor(p)})
        node_bernoulli = BaseCondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs = log_likelihood(node_bernoulli, data)
        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        p = torch.tensor(0.5, requires_grad=True)

        torch_bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(p.grad is not None)

    def test_likelihood_p_0(self):

        # p = 0
        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.0})

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 1.0})

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_marginalization(self):

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": random.random()})
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        p = random.random()
        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[-float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[float("inf")]]),
        )

        # check valid integers inside valid range
        log_likelihood(bernoulli, torch.tensor([[0.0], [1.0]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, bernoulli, torch.tensor([[-1]]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, torch.tensor([[2]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, bernoulli, torch.tensor([[0.5]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
