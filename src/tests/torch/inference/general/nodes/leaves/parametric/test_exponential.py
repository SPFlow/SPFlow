import random
import unittest

import numpy as np
import torch
from packaging import version

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Exponential as BaseExponential
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
from spflow.torch.structure.spn import Exponential


class TestExponential(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        l = random.random() + 1e-7  # small offset to avoid zero

        torch_exponential = Exponential(Scope([0]), l)
        node_exponential = BaseExponential(Scope([0]), l)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_exponential, data)
        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        l = random.random()

        torch_exponential = Exponential(Scope([0]), l)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_exponential.l_aux.grad is not None)

        l_aux_orig = torch_exponential.l_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                l_aux_orig - torch_exponential.l_aux.grad,
                torch_exponential.l_aux,
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(torch_exponential.l, torch_exponential.dist.rate)
        )

    def test_gradient_optimization(self):

        # initialize distribution
        torch_exponential = Exponential(Scope([0]), l=0.5)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Exponential(rate=1.5).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_exponential, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_exponential.l, torch.tensor(1.5), atol=1e-3, rtol=0.3
            )
        )

    def test_likelihood_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Exponential distribution: floats [0,inf) (note: 0 excluded in pytorch support)

        l = 1.5
        exponential = Exponential(Scope([0]), l)

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
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]
            ),
        )
        log_likelihood(exponential, torch.tensor([[10.5]]))

        # check invalid float values (outside range)
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]
            ),
        )

        if version.parse(torch.__version__) < version.parse("1.11.0"):
            # edge case 0 (part of the support in scipy, but NOT pytorch)
            self.assertRaises(
                ValueError, log_likelihood, exponential, torch.tensor([[0.0]])
            )
        else:
            # edge case 0
            log_likelihood(exponential, torch.tensor([[0.0]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
