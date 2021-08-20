from spn.python.structure.nodes.leaves.parametric.parametric import Gaussian
from spn.python.inference.nodes.node import log_likelihood
from spn.torch.structure.nodes.leaves.parametric import TorchGaussian
from spn.torch.inference import log_likelihood

import torch
import numpy as np

import random

import unittest


class TestTorchParametricLeaf(unittest.TestCase):
    def test_gaussian(self):

        mean = random.random()
        stdev = random.random()

        torch_gaussian = TorchGaussian([0], mean, stdev)
        node_gaussian = Gaussian([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs = log_likelihood(node_gaussian, data)
        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gaussian.mean.grad is not None)
        self.assertTrue(torch_gaussian.stdev.grad is not None)

        mean_orig = torch_gaussian.mean.detach().clone()
        stdev_orig = torch_gaussian.stdev.detach().clone()

        optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(mean_orig - torch_gaussian.mean.grad, torch_gaussian.mean))
        self.assertTrue(
            torch.allclose(stdev_orig - torch_gaussian.stdev.grad, torch_gaussian.stdev)
        )

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_gaussian.mean, torch_gaussian.dist.mean))
        self.assertTrue(torch.allclose(torch_gaussian.stdev, torch_gaussian.dist.stddev))


if __name__ == "__main__":
    unittest.main()
