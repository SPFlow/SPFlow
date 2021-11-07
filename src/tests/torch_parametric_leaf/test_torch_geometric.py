from spflow.base.structure.nodes.leaves.parametric import Geometric
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchGeometric, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchGeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        p = random.random()

        torch_geometric = TorchGeometric([0], p)
        node_geometric = Geometric([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs = log_likelihood(SPN(), node_geometric, data)
        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        p = random.random()

        torch_geometric = TorchGeometric([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_geometric.p_aux.grad is not None)

        p_aux_orig = torch_geometric.p_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(p_aux_orig - torch_geometric.p_aux.grad, torch_geometric.p_aux)
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_geometric.p, torch_geometric.dist.probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_geometric = TorchGeometric([0], 0.3)

        # create dummy data
        p_target = 0.8
        data = torch.distributions.Geometric(p_target).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=0.9, momentum=0.6)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_geometric, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(torch_geometric.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3)
        )

    def test_base_backend_conversion(self):

        p = random.random()

        torch_geometric = TorchGeometric([0], p)
        node_geometric = Geometric([0], p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_geometric.get_params()]),
                np.array([*toNodes(torch_geometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_geometric.get_params()]),
                np.array([*toTorch(node_geometric).get_params()]),
            )
        )

    def test_initialiation(self):

        # p = 0
        self.assertRaises(Exception, TorchGeometric, [0], 0.0)
        self.assertRaises(Exception, TorchGeometric, [0], np.inf)
        self.assertRaises(Exception, TorchGeometric, [0], np.nan)

    def test_support(self):

        p = 0.8

        geometric = TorchGeometric([0], p)

        # create test inputs/outputs
        data = torch.tensor(
            [[0], [torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))], [1.5], [1]]
        )

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.all(probs[:3] == 0))
        self.assertTrue(torch.all(probs[-1] != 0))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
