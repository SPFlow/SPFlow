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

        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))


if __name__ == "__main__":
    unittest.main()
