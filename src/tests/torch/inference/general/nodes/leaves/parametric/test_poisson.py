from spflow.meta.data import Scope
from spflow.base.structure.spn import Poisson as BasePoisson
from spflow.base.inference import log_likelihood, likelihood
from spflow.torch.structure.spn import Poisson
from spflow.torch.inference import log_likelihood, likelihood

import torch
import numpy as np
import unittest
import random


class TestPoisson(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        l = random.randint(1, 10)

        torch_poisson = Poisson(Scope([0]), l)
        node_poisson = BasePoisson(Scope([0]), l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data)
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        l = random.randint(1, 10)

        torch_poisson = Poisson(Scope([0]), l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_poisson.l_aux.grad is not None)

        l_aux_orig = torch_poisson.l_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(torch_poisson.l, torch_poisson.dist.rate)
        )

    def test_gradient_optimization(self):

        # initialize distribution
        torch_poisson = Poisson(Scope([0]), l=1.0)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Poisson(rate=4.0).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=0.1)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_poisson, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_poisson.l, torch.tensor(4.0), atol=1e-3, rtol=0.3
            )
        )

    def test_likelihood_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Poisson distribution: integers N U {0}

        l = random.random()

        poisson = Poisson(Scope([0]), l)

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
