#from spflow.base.sampling.sampling_context import SamplingContext
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import NegativeBinomial as BaseNegativeBinomial
from spflow.base.inference.nodes.leaves.parametric.negative_binomial import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.negative_binomial import NegativeBinomial, toBase, toTorch
from spflow.torch.inference.nodes.leaves.parametric.negative_binomial import log_likelihood
from spflow.torch.inference.module import likelihood
#from spflow.torch.sampling import sample

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

    def test_inference(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = NegativeBinomial(Scope([0]), n, p)
        node_negative_binomial = BaseNegativeBinomial(Scope([0]), n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_negative_binomial, data)
        log_probs_torch = log_likelihood(torch_negative_binomial, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = NegativeBinomial(Scope([0]), n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs_torch = log_likelihood(torch_negative_binomial, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_negative_binomial.n.grad is None)
        self.assertTrue(torch_negative_binomial.p_aux.grad is not None)

        n_orig = torch_negative_binomial.n.detach().clone()
        p_aux_orig = torch_negative_binomial.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(n_orig, torch_negative_binomial.n))
        self.assertTrue(
            torch.allclose(
                p_aux_orig - torch_negative_binomial.p_aux.grad, torch_negative_binomial.p_aux
            )
        )

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_negative_binomial = NegativeBinomial(Scope([0]), 5, 0.3)

        # create dummy data
        p_target = 0.8
        data = torch.distributions.NegativeBinomial(5, 1 - p_target).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_negative_binomial, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_negative_binomial.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3)
        )

    def test_likelihood_p_1(self):

        # p = 1
        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_n_float(self):

        negative_binomial = NegativeBinomial(Scope([0]), 1, 0.5)
        self.assertRaises(Exception, likelihood, negative_binomial, 0.5)

    def test_likelihood_marginalization(self):

        negative_binomial = NegativeBinomial(Scope([0]), 20, 0.3)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Negative Binomial distribution: integers N U {0}

        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial(Scope([0]), n, p)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, negative_binomial, torch.tensor([[-float("inf")]])
        )
        self.assertRaises(
            ValueError, log_likelihood, negative_binomial, torch.tensor([[float("inf")]])
        )

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, negative_binomial, torch.tensor([[-1]]))

        # check valid integers within valid range
        log_likelihood(negative_binomial, torch.tensor([[0]]))
        log_likelihood(negative_binomial, torch.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, negative_binomial, torch.tensor([[10.1]]))

"""
    def test_sampling(self):

        # ----- n = 1, p = 1.0 -----

        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(negative_binomial, data, ll_cache={}, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~samples.isnan()] == 0.0))

        # ----- n = 10, p = 0.3 -----

        negative_binomial = NegativeBinomial(Scope([0]), 10, 0.3)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(10 * (1 - 0.3) / 0.3), rtol=0.1))

        # ----- n = 5, p = 0.8 -----

        negative_binomial = NegativeBinomial(Scope([0]), 5, 0.8)

        samples = sample(negative_binomial, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(5 * (1 - 0.8) / 0.8), rtol=0.1))
"""

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
