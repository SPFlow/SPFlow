from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import CondBinomial as BaseCondBinomial
from spflow.base.inference.nodes.leaves.parametric.cond_binomial import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.cond_binomial import CondBinomial, toBase, toTorch
from spflow.torch.inference.nodes.leaves.parametric.cond_binomial import log_likelihood
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestBinomial(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood_module_cond_f(self):

        p = random.random()
        cond_f = lambda data: {'p': p}

        binomial = CondBinomial(Scope([0]), n=1, cond_f=cond_f)

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_args_p(self):

        binomial = CondBinomial(Scope([0]), n=1)

        p = random.random()
        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {'p': p}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_args_cond_f(self):

        binomial = CondBinomial(Scope([0]), n=1)

        p = random.random()
        cond_f = lambda data: {'p': p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {'cond_f': cond_f}

        # create test inputs/outputs
        data = torch.tensor([[0], [1]])
        targets = torch.tensor([[1.0 - p], [p]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_inference(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = CondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})
        node_binomial = BaseCondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_binomial, data)
        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        n = random.randint(2, 10)
        p = torch.tensor(random.random(), requires_grad=True)

        torch_binomial = CondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_binomial.n.grad is None)
        self.assertTrue(p.grad is not None)

    def test_likelihood_p_0(self):

        # p = 0
        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': 0.0})

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_p_1(self):
        
        # p = 1
        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': 1.0})

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_n_0(self):

        # n = 0
        binomial = CondBinomial(Scope([0]), 0, cond_f=lambda data: {'p': 0.5})

        data = torch.tensor([[0.0]])
        targets = torch.tensor([[1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))
    
    def test_likelihood_marginalization(self):

        binomial = CondBinomial(Scope([0]), 5, cond_f=lambda data: {'p': 0.5})
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        binomial = CondBinomial(Scope([0]), 2, cond_f=lambda data: {'p': 0.5})

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[np.inf]]))

        # check valid integers inside valid range
        log_likelihood(binomial, torch.unsqueeze(torch.FloatTensor(list(range(binomial.n + 1))), 1))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-1]]))
        self.assertRaises(
            ValueError, log_likelihood, binomial, torch.tensor([[float(binomial.n + 1)]])
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(binomial.n)), torch.tensor(float(binomial.n + 1))
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(float(binomial.n)), torch.tensor(0.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[0.5]]))
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[3.5]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
