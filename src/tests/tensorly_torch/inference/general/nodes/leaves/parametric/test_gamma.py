import random
import unittest

import numpy as np
import torch
from packaging import version

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Gamma as BaseGamma
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
#from spflow.torch.structure.spn import Gamma
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gamma import Gamma


class TestGamma(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = Gamma(Scope([0]), alpha, beta)
        node_gamma = BaseGamma(Scope([0]), alpha, beta)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_gamma, data)
        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = Gamma(Scope([0]), alpha, beta)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gamma.alpha_aux.grad is not None)
        self.assertTrue(torch_gamma.beta_aux.grad is not None)

        alpha_aux_orig = torch_gamma.alpha_aux.detach().clone()
        beta_aux_orig = torch_gamma.beta_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                alpha_aux_orig - torch_gamma.alpha_aux.grad,
                torch_gamma.alpha_aux,
            )
        )
        self.assertTrue(torch.allclose(beta_aux_orig - torch_gamma.beta_aux.grad, torch_gamma.beta_aux))

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_gamma.alpha, torch_gamma.dist.concentration))
        self.assertTrue(torch.allclose(torch_gamma.beta, torch_gamma.dist.rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_gamma = Gamma(Scope([0]), alpha=1.0, beta=2.0)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Gamma(concentration=2.0, rate=1.0).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=0.5, momentum=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_gamma, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(torch.allclose(torch_gamma.alpha, torch.tensor(2.0), atol=1e-3, rtol=0.3))
        self.assertTrue(torch.allclose(torch_gamma.beta, torch.tensor(1.0), atol=1e-3, rtol=0.3))

    def test_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(gamma, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Gamma distribution: floats (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)

        gamma = Gamma(Scope([0]), 1.0, 1.0)

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
