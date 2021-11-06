from spflow.base.structure.nodes.leaves.parametric import Hypergeometric
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchHypergeometric, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest

class TestTorchHypergeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)
    
    def test_inference(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = TorchHypergeometric([0], N, M, n)
        node_hypergeometric = Hypergeometric([0], N, M, n)

        # create dummy input data (batch size x random variables)
        data = np.array([[4], [5], [10], [11]])

        log_probs = log_likelihood(SPN(), node_hypergeometric, data)
        log_probs_torch = log_likelihood(torch_hypergeometric, torch.tensor(data))

        # TODO: support is handled differently (in log space): -inf for torch and np.finfo().min for numpy (decide how to handle)
        log_probs[log_probs == np.finfo(log_probs.dtype).min] = -np.inf

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = TorchHypergeometric([0], N, M, n)

        # create dummy input data (batch size x random variables)
        data = np.array([[4], [5], [10], [11]])

        log_probs_torch = log_likelihood(torch_hypergeometric, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(4, 1)
        targets_torch.requires_grad = True

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_hypergeometric.N.grad is None)
        self.assertTrue(torch_hypergeometric.M.grad is None)
        self.assertTrue(torch_hypergeometric.n.grad is None)

        # make sure distribution has no (learnable) parameters
        self.assertFalse(list(torch_hypergeometric.parameters()))

    def test_base_backend_conversion(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = TorchHypergeometric([0], N, M, n)
        node_hypergeometric = Hypergeometric([0], N, M, n)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_hypergeometric.get_params()]),
                np.array([*toNodes(torch_hypergeometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_hypergeometric.get_params()]),
                np.array([*toTorch(node_hypergeometric).get_params()]),
            )
        )

    def test_initialization(self):

        self.assertRaises(Exception, TorchHypergeometric, -1, 1, 1)
        self.assertRaises(Exception, TorchHypergeometric, 1, -1, 1)
        self.assertRaises(Exception, TorchHypergeometric, 1, 2, 1)
        self.assertRaises(Exception, TorchHypergeometric, 1, 1, -1)
        self.assertRaises(Exception, TorchHypergeometric, 1, 1, 2)
        self.assertRaises(Exception, TorchHypergeometric, [0], np.inf, 1, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], np.nan, 1, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, np.inf, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, np.nan, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, np.inf)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, np.nan)

    def test_support(self):

        N = 15
        M = 10
        n = 10

        hypergeometric = TorchHypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = torch.tensor([[4], [11], [5], [10]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs[:2] == 0))
        self.assertTrue(all(probs[2:] != 0))

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
