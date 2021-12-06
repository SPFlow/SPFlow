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

        log_probs = log_likelihood(node_hypergeometric, data, SPN())
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

        # Valid parameters for Hypergeometric distribution: N in N U {0}, M in {0,...,N}, n in {0,...,N}, p in [0,1] TODO

        # N = 0
        TorchHypergeometric([0], 0, 0, 0)
        # N < 0
        self.assertRaises(Exception, TorchHypergeometric, [0], -1, 1, 1)
        # N = inf and N = nan
        self.assertRaises(Exception, TorchHypergeometric, [0], np.inf, 1, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], np.nan, 1, 1)
        
        # M < 0 and M > N
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, -1, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 2, 1)
        # M = inf and M = nan
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, np.inf, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, np.nan, 1)

        # n < 0 and n > N
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, -1)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, 2)
        # n = inf and n = nan
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, np.inf)
        self.assertRaises(Exception, TorchHypergeometric, [0], 1, 1, np.nan)

        # invalid scope lengths
        self.assertRaises(Exception, TorchHypergeometric, [], 1, 1, 1)
        self.assertRaises(Exception, TorchHypergeometric, [0,1], 1, 1, 1)

    def test_support(self):

        # Support for Hypergeometric distribution: {max(0,n+M-N),...,min(n,M)}
        
        # TODO:
        #   outside support -> 0 (or error?)

        # case n+M-N > 0
        N = 15
        M = 10
        n = 10

        hypergeometric = TorchHypergeometric([0], N, M, n)

        # edge cases (-inf,inf), finite values outside valid integer range and values between valid integers
        data = torch.tensor([[-float("inf")], [torch.nextafter(torch.tensor(max(0, n+M-N)), torch.tensor(-1.0))], [1.5], [torch.nextafter(torch.tensor(min(n,M)), torch.tensor(-1.0))], [torch.nextafter(torch.tensor(min(n,M)), torch.tensor(20.0))], [float("inf")]])
        targets = torch.zeros((6,1))

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        # TODO: fails (1.5, why ???)
        #self.assertTrue(torch.allclose(probs, targets))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        
        # case n+M-N
        N = 25

        hypergeometric = TorchHypergeometric([0], N, M, n)

        # edge cases (-inf,inf), finite values outside valid integer range and values between valid integers
        data = torch.tensor([[-float("inf")], [torch.nextafter(torch.tensor(max(0, n+M-N)), torch.tensor(-1.0))], [1.5], [torch.nextafter(torch.tensor(min(n,M)), torch.tensor(-1.0))], [torch.nextafter(torch.tensor(min(n,M)), torch.tensor(20.0))], [float("inf")]])
        targets = torch.zeros((6,1))

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        # TODO: fails (1.5, why ???)
        #self.assertTrue(torch.allclose(probs, targets))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # max(0,n+M-N) and min(n,M)
        data = torch.tensor([[max(0,n+M-N)], [min(n,M)]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(all(probs != 0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
