from spflow.base.structure.nodes.leaves.parametric import Gamma
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchGamma, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchGamma(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = TorchGamma([0], alpha, beta)
        node_gamma = Gamma([0], alpha, beta)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_gamma, data, SPN())
        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = TorchGamma([0], alpha, beta)

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
            torch.allclose(alpha_aux_orig - torch_gamma.alpha_aux.grad, torch_gamma.alpha_aux)
        )
        self.assertTrue(
            torch.allclose(beta_aux_orig - torch_gamma.beta_aux.grad, torch_gamma.beta_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_gamma.alpha, torch_gamma.dist.concentration))
        self.assertTrue(torch.allclose(torch_gamma.beta, torch_gamma.dist.rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_gamma = TorchGamma([0], alpha=1.0, beta=2.0)

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

    def test_base_backend_conversion(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = TorchGamma([0], alpha, beta)
        node_gamma = Gamma([0], alpha, beta)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gamma.get_params()]),
                np.array([*toNodes(torch_gamma).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gamma.get_params()]), np.array([*toTorch(node_gamma).get_params()])
            )
        )

    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        TorchGamma([0], torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)), 1.0)
        TorchGamma([0], 1.0, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))

        # alpha < 0
        self.assertRaises(Exception, TorchGamma, [0], np.nextafter(0.0, -1.0), 1.0)
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, TorchGamma, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchGamma, [0], np.nan, 1.0)
        
        # beta = < 0
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.nextafter(0.0, -1.0))
        # beta = inf and beta = non
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.inf)
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.nan)

        # invalid scope lengths
        self.assertRaises(Exception, TorchGamma, [], 1.0, 1.0)
        self.assertRaises(Exception, TorchGamma, [0,1], 1.0, 1.0)

    def test_support(self):
        
        # Support for Gamma distribution: (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)
        #
        #   outside support -> nan (or 0?)

        gamma = TorchGamma([0], 1.0, 1.0)

        # edge cases (-inf,inf) and finite values < 0
        data = torch.tensor([[-float("inf")], [torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))], [float("inf")]])
        targets = torch.zeros((3,1))

        # TODO: fails (support, which one?)
        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(torch.allclose(probs, targets))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # finite values > 0
        data =  torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(all(data != 0.0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # TODO: 0


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
