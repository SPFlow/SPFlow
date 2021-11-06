from spflow.base.structure.nodes.leaves.parametric import Binomial
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchBinomial, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest

class TestTorchBinomial(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)
    
    def test_inference(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = TorchBinomial([0], n, p)
        node_binomial = Binomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(SPN(), node_binomial, data)
        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = TorchBinomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_binomial.n.grad is None)
        self.assertTrue(torch_binomial.p_aux.grad is not None)

        n_orig = torch_binomial.n.detach().clone()
        p_aux_orig = torch_binomial.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_binomial.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(n_orig, torch_binomial.n))
        self.assertTrue(
            torch.allclose(p_aux_orig - torch_binomial.p_aux.grad, torch_binomial.p_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.equal(torch_binomial.n, torch_binomial.dist.total_count.long()))
        self.assertTrue(torch.allclose(torch_binomial.p, torch_binomial.dist.probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_binomial = TorchBinomial([0], 5, 0.3)

        # create dummy data
        p_target = 0.8
        data = torch.distributions.Binomial(5, p_target).sample((100000,1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_binomial.parameters(), lr=0.5)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()
            
            # compute negative log-likelihood
            nll = -log_likelihood(torch_binomial, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(torch.allclose(torch_binomial.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3))

    def test_base_backend_conversion(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = TorchBinomial([0], n, p)
        node_binomial = Binomial([0], n, p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_binomial.get_params()]),
                np.array([*toNodes(torch_binomial).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_binomial.get_params()]),
                np.array([*toTorch(node_binomial).get_params()]),
            )
        )

    def test_initialization(self):

        # p = 0
        binomial = TorchBinomial([0], 1, 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # TODO (fails): self.assertTrue(torch.allclose(probs, targets))

        # p = 1
        binomial = TorchBinomial([0], 1, 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # TODO (fails): self.assertTrue(torch.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(
            Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
        )
        self.assertRaises(
            Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))
        )

        # n = 0
        # TODO (not supported): binomial = TorchBinomial([0], 0, 0.5)

        # data = torch.tensor([[0.0], [1.0]])
        # targets = torch.tensor([[1.0], [0.0]])

        # probs = likelihood(binomial, data)
        # log_probs = log_likelihood(binomial, data)

        # self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # self.assertTrue(torch.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, TorchBinomial, [0], -1, 0.5)

        # TODO: n float

        # inf, nan
        self.assertRaises(Exception, TorchBinomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0], np.nan, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.nan)

    def test_support(self):

        binomial = TorchBinomial([0], 1, 0.0)

        data = torch.tensor([[-1.0], [2.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs == 0))

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
