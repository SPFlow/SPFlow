from spflow.base.structure.nodes.leaves.parametric import Exponential
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchExponential, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest

class TestTorchExponential(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)
    
    def test_inference(self):

        l = random.random()

        torch_exponential = TorchExponential([0], l)
        node_exponential = Exponential([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(SPN(), node_exponential, data)
        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        l = random.random()

        torch_exponential = TorchExponential([0], l)

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
            torch.allclose(l_aux_orig - torch_exponential.l_aux.grad, torch_exponential.l_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist.rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_exponential = TorchExponential([0], l=0.5)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Exponential(rate=1.5).sample((100000,1))

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

        self.assertTrue(torch.allclose(torch_exponential.l, torch.tensor(1.5), atol=1e-3, rtol=0.3))

    def test_base_backend_conversion(self):

        l = random.random()

        torch_exponential = TorchExponential([0], l)
        node_exponential = Exponential([0], l)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_exponential.get_params()]),
                np.array([*toNodes(torch_exponential).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_exponential.get_params()]),
                np.array([*toTorch(node_exponential).get_params()]),
            )
        )

    def test_initialization(self):

        TorchExponential([0], torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
        self.assertRaises(Exception, TorchExponential, [0], 0.0)
        self.assertRaises(Exception, TorchExponential, [0], -1.0)
        self.assertRaises(Exception, TorchExponential, [0], np.inf)
        self.assertRaises(Exception, TorchExponential, [0], np.nan)

    def test_support(self):

        l = 1.5

        exponential = TorchExponential([0], l)

        # create test inputs/outputs
        data = torch.tensor(
            [[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]
        )  # TODO (fails):, [0.0]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        # TODO (fails): self.assertTrue(all(probs[1] != 0.0))

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
