import random
import unittest

import numpy as np
import torch

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import LogNormal as BaseLogNormal
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
#from spflow.torch.structure.spn import LogNormal
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_log_normal import LogNormal


class TestLogNormal(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_log_normal = LogNormal(Scope([0]), mean, std)
        node_log_normal = BaseLogNormal(Scope([0]), mean, std)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_log_normal, data)
        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_log_normal = LogNormal(Scope([0]), mean, std)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_log_normal.mean.grad is not None)
        self.assertTrue(torch_log_normal.std_aux.grad is not None)

        mean_orig = torch_log_normal.mean.detach().clone()
        std_aux_orig = torch_log_normal.std_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(mean_orig - torch_log_normal.mean.grad, torch_log_normal.mean))
        self.assertTrue(
            torch.allclose(
                std_aux_orig - torch_log_normal.std_aux.grad,
                torch_log_normal.std_aux,
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_log_normal.mean, torch_log_normal.dist.loc))
        self.assertTrue(torch.allclose(torch_log_normal.std, torch_log_normal.dist.scale))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_log_normal = LogNormal(Scope([0]), mean=1.0, std=2.0)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.LogNormal(0.0, 1.0).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=0.5, momentum=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_log_normal, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(torch.allclose(torch_log_normal.mean, torch.tensor(0.0), atol=1e-3, rtol=0.3))
        self.assertTrue(torch.allclose(torch_log_normal.std, torch.tensor(1.0), atol=1e-3, rtol=0.3))

    def test_likelihood_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(log_normal, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Log-Normal distribution: floats (0,inf)

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            log_normal,
            torch.tensor([[float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            log_normal,
            torch.tensor([[-float("inf")]]),
        )

        # invalid float values
        self.assertRaises(ValueError, log_likelihood, log_normal, torch.tensor([[0]]))

        # valid float values
        log_likelihood(
            log_normal,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        log_likelihood(log_normal, torch.tensor([[4.3]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
