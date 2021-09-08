from spn.python.structure.nodes.leaves.parametric.parametric import (
    Gaussian,
    LogNormal,
    MultivariateGaussian,
    Uniform,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
    Geometric,
    Exponential,
    Gamma,
)
from spn.python.inference.nodes.node import log_likelihood
from spn.torch.structure.nodes.leaves.parametric import (
    toNodes,
    toTorch,
    TorchGaussian,
    TorchLogNormal,
    TorchMultivariateGaussian,
    TorchUniform,
    TorchBernoulli,
    TorchBinomial,
    TorchNegativeBinomial,
    TorchPoisson,
    TorchGeometric,
    TorchExponential,
    TorchGamma,
)
from spn.torch.inference import log_likelihood

import torch
import numpy as np

import random

import unittest


class TestTorchParametricLeaf(unittest.TestCase):
    def test_gaussian(self):

        # ----- check inference -----

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

        # ----- check gradient computation -----

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

        # reset torch distribution after gradient update
        torch_gaussian = TorchGaussian([0], mean, stdev)

        # ----- check conversion between python and backend -----

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

    def test_log_normal(self):

        # ----- check inference -----

        mean = random.random()
        stdev = random.random()

        torch_log_normal = TorchLogNormal([0], mean, stdev)
        node_log_normal = LogNormal([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_log_normal, data)
        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_log_normal.mean.grad is not None)
        self.assertTrue(torch_log_normal.stdev.grad is not None)

        mean_orig = torch_log_normal.mean.detach().clone()
        stdev_orig = torch_log_normal.stdev.detach().clone()

        optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(mean_orig - torch_log_normal.mean.grad, torch_log_normal.mean)
        )
        self.assertTrue(
            torch.allclose(stdev_orig - torch_log_normal.stdev.grad, torch_log_normal.stdev)
        )

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_log_normal.mean, torch_log_normal.dist.loc))
        self.assertTrue(torch.allclose(torch_log_normal.stdev, torch_log_normal.dist.scale))

        # reset torch distribution after gradient update
        torch_log_normal = TorchLogNormal([0], mean, stdev)

        # ----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_log_normal.get_params()]),
                np.array([*toNodes(torch_log_normal).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_log_normal.get_params()]),
                np.array([*toTorch(node_log_normal).get_params()]),
            )
        )

    def test_multivariate_gaussian(self):
        pass

    def test_uniform(self):

        # ----- check inference -----

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = Uniform([0], start, end)
        torch_uniform = TorchUniform([0], start, end)

        # create test inputs/outputs
        data = np.random.rand(3, 1) * 2.0

        log_probs = log_likelihood(node_uniform, data)
        log_probs_torch = log_likelihood(torch_uniform, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)
        targets_torch.requires_grad = True

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_uniform.start.grad is None)
        self.assertTrue(torch_uniform.end.grad is None)

        # make sure distribution has no (learnable) parameters
        self.assertFalse(list(torch_uniform.parameters()))

        # ----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_uniform.get_params()]),
                np.array([*toNodes(torch_uniform).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_uniform.get_params()]),
                np.array([*toTorch(node_uniform).get_params()]),
            )
        )

    def test_bernoulli(self):

        # ----- check inference -----

        p = random.random()

        torch_bernoulli = TorchBernoulli([0], p)
        node_bernoulli = Bernoulli([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs = log_likelihood(node_bernoulli, data)
        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_bernoulli.p.grad is not None)

        p_orig = torch_bernoulli.p.detach().clone()

        optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(p_orig - torch_bernoulli.p.grad, torch_bernoulli.p))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_bernoulli.p, torch_bernoulli.dist.probs))

        # reset torch distribution after gradient update
        torch_bernoulli = TorchBernoulli([0], p)

        # ----- check conversion between python and backend -----

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

    def test_binomial(self):

        # ----- check inference -----

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = TorchBinomial([0], n, p)
        node_binomial = Binomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_binomial, data)
        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_binomial.n.grad is None)
        self.assertTrue(torch_binomial.p.grad is not None)

        n_orig = torch_binomial.n.detach().clone()
        p_orig = torch_binomial.p.detach().clone()

        optimizer = torch.optim.SGD(torch_binomial.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(n_orig, torch_binomial.n))
        self.assertTrue(torch.allclose(p_orig - torch_binomial.p.grad, torch_binomial.p))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.equal(torch_binomial.n, torch_binomial.dist.total_count.long()))
        self.assertTrue(torch.allclose(torch_binomial.p, torch_binomial.dist.probs))

        # reset torch distribution after gradient update
        torch_binomial = TorchBinomial([0], n, p)

        # ----- check conversion between python and backend -----

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

    def test_negative_binomial(self):

        # ----- check inference -----

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = TorchNegativeBinomial([0], n, p)
        node_negative_binomial = NegativeBinomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_negative_binomial, data)
        log_probs_torch = log_likelihood(
            torch_negative_binomial, torch.tensor(data, dtype=torch.float32)
        )

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_negative_binomial.n.grad is None)
        self.assertTrue(torch_negative_binomial.p.grad is not None)

        n_orig = torch_negative_binomial.n.detach().clone()
        p_orig = torch_negative_binomial.p.detach().clone()

        optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(n_orig, torch_negative_binomial.n))
        self.assertTrue(
            torch.allclose(p_orig - torch_negative_binomial.p.grad, torch_negative_binomial.p)
        )

        # reset torch distribution after gradient update
        torch_negative_binomial = TorchNegativeBinomial([0], n, p)

        # ----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_negative_binomial.get_params()]),
                np.array([*toNodes(torch_negative_binomial).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_negative_binomial.get_params()]),
                np.array([*toTorch(node_negative_binomial).get_params()]),
            )
        )

    def test_poisson(self):

        # ----- check inference -----

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data)
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_poisson.l.grad is not None)

        l_orig = torch_poisson.l.detach().clone()

        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(l_orig - torch_poisson.l.grad, torch_poisson.l))
        self.assertTrue(torch.allclose(l_orig - torch_poisson.l.grad, torch_poisson.l))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_poisson.l, torch_poisson.dist.rate))

        # reset torch distribution after gradient update
        torch_poisson = TorchPoisson([0], l)

        # ----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_poisson.get_params()]),
                np.array([*toNodes(torch_poisson).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_poisson.get_params()]),
                np.array([*toTorch(node_poisson).get_params()]),
            )
        )

    def test_geometric(self):

        # ----- check inference -----

        p = random.random()

        torch_geometric = TorchGeometric([0], p)
        node_geometric = Geometric([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs = log_likelihood(node_geometric, data)
        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_geometric.p.grad is not None)

        p_orig = torch_geometric.p.detach().clone()

        optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(p_orig - torch_geometric.p.grad, torch_geometric.p))
        self.assertTrue(torch.allclose(p_orig - torch_geometric.p.grad, torch_geometric.p))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_geometric.p, torch_geometric.dist.probs))

        # reset torch distribution after gradient update
        torch_geometric = TorchGeometric([0], p)

        # ----- check conversion between python and backend -----

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

    def test_hypergeometri(self):
        pass

    def test_exponential(self):

        # ----- check inference -----

        l = random.random()

        torch_exponential = TorchExponential([0], l)
        node_exponential = Exponential([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_exponential, data)
        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_exponential.l.grad is not None)

        l_orig = torch_exponential.l.detach().clone()

        optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(l_orig - torch_exponential.l.grad, torch_exponential.l))
        self.assertTrue(torch.allclose(l_orig - torch_exponential.l.grad, torch_exponential.l))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist.rate))

        # reset torch distribution after gradient update
        torch_exponential = TorchExponential([0], l)

        # ----- check conversion between python and backend -----

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

    def test_gamma(self):
        pass
        """
        #----- check inference -----

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = TorchGamma([0], alpha, beta)
        node_gamma = Gamma([0], alpha, beta)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_gamma, data)
        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data, dtype=torch.float32))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        #----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gamma.l.grad is not None)

        alpha_orig = torch_gamma.alpha.detach().clone()

        optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(l_orig - torch_gamma.l.grad, torch_gamma.l))
        self.assertTrue(
            torch.allclose(l_orig - torch_gamma.l.grad, torch_gamma.l)
        )

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist.rate))

        # reset torch distribution after gradient update
        torch_geometric = TorchGeometric([0], p)
    
        #----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(np.allclose(np.array([*torch_geometric.get_params()]), np.array([*toNodes(torch_geometric).get_params()])))
        # check conversion from python to torch
        self.assertTrue(np.allclose(np.array([*node_geometric.get_params()]), np.array([*toTorch(node_geometric).get_params()])))
        """


if __name__ == "__main__":
    unittest.main()
