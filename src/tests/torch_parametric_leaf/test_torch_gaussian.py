from spflow.base.structure.nodes.leaves.parametric import Gaussian
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchGaussian, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        mean = random.random()
        stdev = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = TorchGaussian([0], mean, stdev)
        node_gaussian = Gaussian([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs = log_likelihood(SPN(), node_gaussian, data)
        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        mean = random.random()
        stdev = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = TorchGaussian([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gaussian.mean.grad is not None)
        self.assertTrue(torch_gaussian.stdev_aux.grad is not None)

        mean_orig = torch_gaussian.mean.detach().clone()
        stdev_aux_orig = torch_gaussian.stdev_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(mean_orig - torch_gaussian.mean.grad, torch_gaussian.mean))
        self.assertTrue(
            torch.allclose(stdev_aux_orig - torch_gaussian.stdev_aux.grad, torch_gaussian.stdev_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_gaussian.mean, torch_gaussian.dist.mean))
        self.assertTrue(torch.allclose(torch_gaussian.stdev, torch_gaussian.dist.stddev))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_gaussian = TorchGaussian([0], mean=1.0, stdev=2.0)

        torch.manual_seed(0)

        # create dummy data (unit variance Gaussian)
        data = torch.randn((100000, 1))
        data = (data - data.mean()) / data.std()

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_gaussian, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_gaussian.mean, torch.tensor(0.0), atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(torch_gaussian.stdev, torch.tensor(1.0), atol=1e-3, rtol=1e-3)
        )

    def test_base_backend_conversion(self):

        mean = random.random()
        stdev = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = TorchGaussian([0], mean, stdev)
        node_gaussian = Gaussian([0], mean, stdev)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gaussian.get_params()]),
                np.array([*toNodes(torch_gaussian).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gaussian.get_params()]),
                np.array([*toTorch(node_gaussian).get_params()]),
            )
        )

    def test_initialization(self):

        mean = random.random()

        self.assertRaises(Exception, TorchGaussian, [0], mean, 0.0)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, TorchGaussian, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchGaussian, [0], np.nan, 1.0)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.inf)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.nan)

        # invalid scope length
        self.assertRaises(Exception, TorchGaussian, [], 0.0, 1.0)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
