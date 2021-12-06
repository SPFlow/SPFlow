from spflow.base.structure.nodes.leaves.parametric import Bernoulli
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchBernoulli, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchBernoulli(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        p = random.random()

        torch_bernoulli = TorchBernoulli([0], p)
        node_bernoulli = Bernoulli([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs = log_likelihood(node_bernoulli, data, SPN())
        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        p = random.random()

        torch_bernoulli = TorchBernoulli([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_bernoulli.p_aux.grad is not None)

        p_aux_orig = torch_bernoulli.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(p_aux_orig - torch_bernoulli.p_aux.grad, torch_bernoulli.p_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_bernoulli.p, torch_bernoulli.dist.probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_bernoulli = TorchBernoulli([0], 0.3)

        # create dummy data
        p_target = 0.8
        data = torch.bernoulli(torch.full((100000, 1), p_target))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=0.5, momentum=0.5)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_bernoulli, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_bernoulli.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3)
        )

    def test_base_backend_conversion(self):

        p = random.random()

        torch_bernoulli = TorchBernoulli([0], p)
        node_bernoulli = Bernoulli([0], p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_bernoulli.get_params()]),
                np.array([*toNodes(torch_bernoulli).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_bernoulli.get_params()]),
                np.array([*toTorch(node_bernoulli).get_params()]),
            )
        )

    def test_initialization(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = TorchBernoulli([0], 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

        # p = 1
        bernoulli = TorchBernoulli([0], 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, TorchBernoulli, [0], torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)))
        self.assertRaises(Exception, TorchBernoulli, [0], torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)))

        # inf, nan
        self.assertRaises(Exception, TorchBernoulli, [0], np.inf)
        self.assertRaises(Exception, TorchBernoulli, [0], np.nan)

        # invalid scope lengths
        self.assertRaises(Exception, TorchBernoulli, [], 0.5)
        self.assertRaises(Exception, TorchBernoulli, [0,1], 0.5)

    def test_support(self):

        # Support for Bernoulli distribution: {0,1}
    
        # TODO:
        #   outside support -> 0 (or error?)

        p = random.random()

        bernoulli = TorchBernoulli([0], p)

        # edge cases (-inf,inf), finite values outside [0,1] and values within (0,1)
        data = torch.tensor([[-float("inf")], [-1.0], [torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))], [0.5], [torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))], [2.0], [float("inf")]])
        targets = torch.zeros((7,1))
        
        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, targets))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
