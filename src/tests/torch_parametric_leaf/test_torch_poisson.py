from spflow.base.structure.nodes.leaves.parametric import Poisson
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchPoisson, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchPoisson(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data, SPN())
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)

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
        self.assertTrue(torch.allclose(l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux))

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_poisson.l, torch_poisson.dist.rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_poisson = TorchPoisson([0], l=1.0)

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

        self.assertTrue(torch.allclose(torch_poisson.l, torch.tensor(4.0), atol=1e-3, rtol=0.3))

    def test_base_backend_conversion(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_poisson.get_params()]),
                np.array([*toNodes(torch_poisson).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_poisson.get_params()]),
                np.array([*toTorch(node_poisson).get_params()]),
            )
        )

    def test_initialization(self):

        self.assertRaises(Exception, TorchPoisson, [0], -np.inf)
        self.assertRaises(Exception, TorchPoisson, [0], np.inf)
        self.assertRaises(Exception, TorchPoisson, [0], np.nan)

        # invalid scope length
        self.assertRaises(Exception, TorchPoisson, [], 1)

    def test_support(self):

        l = random.random()

        poisson = TorchPoisson([0], l)

        # create test inputs/outputs
        data = torch.tensor([[-1.0], [-0.5], [0.0]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.all(probs[:2] == 0))
        self.assertTrue(torch.all(probs[-1] != 0))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
