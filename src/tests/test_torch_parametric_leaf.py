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
    Hypergeometric,
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
    TorchHypergeometric,
    TorchExponential,
    TorchGamma,
)
from spn.torch.inference import log_likelihood, likelihood

import torch
import numpy as np

import random

import unittest


class TestTorchParametricLeaf(unittest.TestCase):
    def test_gaussian(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        mean = random.random()
        stdev = random.random()

        torch_gaussian = TorchGaussian([0], mean, stdev)
        node_gaussian = Gaussian([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.randn(3, 1)

        log_probs = log_likelihood(node_gaussian, data)
        log_probs_torch = log_likelihood(torch_gaussian, torch.tensor(data))
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

        # ----- invalid parameters -----
        mean = random.random()

        self.assertRaises(Exception, TorchGaussian, [0], mean, 0.0)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, TorchGaussian, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchGaussian, [0], np.nan, 1.0)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.inf)
        self.assertRaises(Exception, TorchGaussian, [0], mean, np.nan)

    def test_log_normal(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        mean = random.random()
        stdev = random.random()

        torch_log_normal = TorchLogNormal([0], mean, stdev)
        node_log_normal = LogNormal([0], mean, stdev)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_log_normal, data)
        log_probs_torch = log_likelihood(torch_log_normal, torch.tensor(data))

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

        # ----- invalid parameters -----
        mean = random.random()

        self.assertRaises(Exception, TorchLogNormal, [0], mean, 0.0)
        self.assertRaises(Exception, TorchLogNormal, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, TorchLogNormal, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchLogNormal, [0], np.nan, 1.0)
        self.assertRaises(Exception, TorchLogNormal, [0], mean, np.inf)
        self.assertRaises(Exception, TorchLogNormal, [0], mean, np.nan)

        # ----- support -----
        mean = 0.0
        stdev = 1.0

        log_normal = TorchLogNormal([0], mean, stdev)

        # create test inputs testing around the boundaries
        data = torch.tensor(
            [[-1.0], [0.0], [float('Inf')], [torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]
        )

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(torch.isinf(log_probs[:3])))
        self.assertTrue(all(~torch.isinf(log_probs[3:])))

    def test_multivariate_gaussian(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        mean_vector = np.arange(3)
        covariance_matrix = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

        torch_multivariate_gaussian = TorchMultivariateGaussian(
            [0, 1, 2], mean_vector, covariance_matrix
        )
        node_multivariate_gaussian = MultivariateGaussian(
            [0, 1, 2], mean_vector.tolist(), covariance_matrix.tolist()
        )

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 3)

        log_probs = log_likelihood(node_multivariate_gaussian, data)
        log_probs_torch = log_likelihood(
            torch_multivariate_gaussian, torch.tensor(data)
        )

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_multivariate_gaussian.mean_vector.grad is not None)
        self.assertTrue(torch_multivariate_gaussian.covariance_matrix.grad is not None)

        mean_vector_orig = torch_multivariate_gaussian.mean_vector.detach().clone()
        covariance_matrix_orig = torch_multivariate_gaussian.covariance_matrix.detach().clone()

        optimizer = torch.optim.SGD(torch_multivariate_gaussian.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                mean_vector_orig - torch_multivariate_gaussian.mean_vector.grad,
                torch_multivariate_gaussian.mean_vector,
            )
        )
        self.assertTrue(
            torch.allclose(
                covariance_matrix_orig - torch_multivariate_gaussian.covariance_matrix.grad,
                torch_multivariate_gaussian.covariance_matrix,
            )
        )

        # TODO: ensure that covariance matrix remains valid

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.mean_vector, torch_multivariate_gaussian.dist.loc
            )
        )
        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.covariance_matrix,
                torch_multivariate_gaussian.dist.covariance_matrix,
            )
        )

        # reset torch distribution after gradient update
        torch_multivariate_gaussian = TorchMultivariateGaussian(
            [0, 1, 2], mean_vector, covariance_matrix
        )

        # ----- check conversion between python and backend -----

        node_params = node_multivariate_gaussian.get_params()
        torch_params = torch_multivariate_gaussian.get_params()

        # check conversion from torch to python
        torch_to_node_params = toNodes(torch_multivariate_gaussian).get_params()

        self.assertTrue(
            np.allclose(
                np.array([torch_params[0]]),
                np.array([torch_to_node_params[0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.array([torch_params[1]]),
                np.array([torch_to_node_params[1]]),
            )
        )
        # check conversion from python to torch#
        node_to_torch_params = toTorch(node_multivariate_gaussian).get_params()

        self.assertTrue(
            np.allclose(
                np.array([node_params[0]]),
                np.array([node_to_torch_params[0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.array([node_params[1]]),
                np.array([node_to_torch_params[1]]),
            )
        )

    def test_uniform(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = Uniform([0], start, end)
        torch_uniform = TorchUniform([0], start, end)

        # create test inputs/outputs
        data_np = np.array(
            [
                [np.nextafter(start, -np.inf)],
                [start],
                [(start + end) / 2.0],
                [end],
                [np.nextafter(end, np.inf)],
            ]
        )
        data_torch = torch.tensor(
            [
                [torch.nextafter(torch.tensor(start), -torch.tensor(float("Inf")))],
                [start],
                [(start + end) / 2.0],
                [end],
                [torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))],
            ]
        )

        log_probs = log_likelihood(node_uniform, data_np)
        log_probs_torch = log_likelihood(torch_uniform, data_torch)

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(5, 1)
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

        # ----- invalid parameters -----
        start_end = random.random()

        self.assertRaises(Exception, TorchUniform, [0], start_end, start_end)
        self.assertRaises(Exception, TorchUniform, [0], start_end, np.nextafter(start_end, -1.0))
        self.assertRaises(Exception, TorchUniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], np.nan, 0.0)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, TorchUniform, [0], 0.0, np.nan)

    def test_bernoulli(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        p = random.random()

        torch_bernoulli = TorchBernoulli([0], p)
        node_bernoulli = Bernoulli([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs = log_likelihood(node_bernoulli, data)
        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

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

        # ----- (invalid) parameters -----

        # p = 0
        bernoulli = TorchBernoulli([0], 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # TODO (fails): self.assertTrue(torch.allclose(probs, targets))

        # p = 1
        bernoulli = TorchBernoulli([0], 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # TODO (fails): self.assertTrue(torch.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, TorchBernoulli, [0], torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)))
        self.assertRaises(Exception, TorchBernoulli, [0], torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)))

        # inf, nan
        self.assertRaises(Exception, TorchBernoulli, [0], np.inf)
        self.assertRaises(Exception, TorchBernoulli, [0], np.nan)

        # ----- support -----
        p = random.random()

        bernoulli = TorchBernoulli([0], p)

        data = torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))], [0.5], [torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs == 0.0))

    def test_binomial(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = TorchBinomial([0], n, p)
        node_binomial = Binomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_binomial, data)
        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

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

        # ----- (invalid) parameters -----

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
        self.assertRaises(Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)))
        self.assertRaises(Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)))

        # n = 0
        # TODO (not supported): binomial = TorchBinomial([0], 0, 0.5)

        #data = torch.tensor([[0.0], [1.0]])
        #targets = torch.tensor([[1.0], [0.0]])

        #probs = likelihood(binomial, data)
        #log_probs = log_likelihood(binomial, data)

        #self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        #self.assertTrue(torch.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, TorchBinomial, [0], -1, 0.5)

        # TODO: n float

        # inf, nan
        self.assertRaises(Exception, TorchBinomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0], np.nan, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.nan)

        # ----- support -----

        binomial = TorchBinomial([0], 1, 0.0)

        data = torch.tensor([[-1.0], [2.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs == 0))

    def test_negative_binomial(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        n = random.randint(2, 10)
        p = random.random()

        torch_negative_binomial = TorchNegativeBinomial([0], n, p)
        node_negative_binomial = NegativeBinomial([0], n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_negative_binomial, data)
        log_probs_torch = log_likelihood(
            torch_negative_binomial, torch.tensor(data)
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

        # ----- (invalid) parameters -----

        # p = 1
        negative_binomial = TorchNegativeBinomial([0], 1, 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        # TODO (fail): self.assertTrue(torch.allclose(probs, targets))

        # p = 0
        self.assertRaises(Exception, TorchNegativeBinomial, [0], 1, 0.0)

        # 1 >= p > 0
        TorchNegativeBinomial([0], 1, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))

        # p < 0 and p > 1
        self.assertRaises(Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)))
        self.assertRaises(Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)))

        # p inf, nan
        self.assertRaises(Exception, TorchNegativeBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, TorchNegativeBinomial, [0], 1, np.nan)

        # n = 0
        TorchNegativeBinomial([0], 0.0, 1.0)

        # n < 0
        self.assertRaises(Exception, TorchNegativeBinomial, [0], torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)), 1.0)

        # n inf, nan
        self.assertRaises(Exception, TorchNegativeBinomial, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchNegativeBinomial, [0], np.nan, 1.0)

        # TODO: n float

        # ----- support -----
        n = 20
        p = 0.3

        negative_binomial = TorchNegativeBinomial([0], n, p)

        data = torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        self.assertTrue(all(probs[1] != 0.0))

    def test_poisson(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data)
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

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

        # ----- invalid parameters -----
        self.assertRaises(Exception, TorchPoisson, [0], -np.inf)
        self.assertRaises(Exception, TorchPoisson, [0], np.inf)
        self.assertRaises(Exception, TorchPoisson, [0], np.nan)

        # ----- support -----

        l = random.random()

        poisson = TorchPoisson([0], l)

        # create test inputs/outputs
        data = torch.tensor([[-1.0], [-0.5], [0.0]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.all(probs[:2] == 0))
        self.assertTrue(torch.all(probs[-1] != 0))

    def test_geometric(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        p = random.random()

        torch_geometric = TorchGeometric([0], p)
        node_geometric = Geometric([0], p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, 10, (3, 1))

        log_probs = log_likelihood(node_geometric, data)
        log_probs_torch = log_likelihood(torch_geometric, torch.tensor(data))

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

        # ----- invalid parameters -----

        # p = 0
        self.assertRaises(Exception, TorchGeometric, [0], 0.0)
        self.assertRaises(Exception, TorchGeometric, [0], np.inf)
        self.assertRaises(Exception, TorchGeometric, [0], np.nan)

        # ----- support -----

        p = 0.8

        geometric = TorchGeometric([0], p)

        # create test inputs/outputs
        data = torch.tensor([[0], [torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))], [1.5], [1]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.all(probs[:3] == 0))
        self.assertTrue(torch.all(probs[-1] != 0))

    def test_hypergeometri(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = TorchHypergeometric([0], N, M, n)
        node_hypergeometric = Hypergeometric([0], N, M, n)

        # create dummy input data (batch size x random variables)
        data = np.array([[4], [5], [10], [11]])

        log_probs = log_likelihood(node_hypergeometric, data)
        log_probs_torch = log_likelihood(
            torch_hypergeometric, torch.tensor(data)
        )

        # TODO: support is handled differently (in log space): -inf for torch and np.finfo().min for numpy (decide how to handle)
        log_probs[log_probs == np.finfo(log_probs.dtype).min] = -np.inf

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

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

        # ----- check conversion between python and backend -----

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

        # ----- invalid parameters -----
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

        # ----- support -----
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

    def test_exponential(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        l = random.random()

        torch_exponential = TorchExponential([0], l)
        node_exponential = Exponential([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_exponential, data)
        log_probs_torch = log_likelihood(torch_exponential, torch.tensor(data))

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

        # ----- (invalid) parameters -----
        TorchExponential([0], torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
        self.assertRaises(Exception, TorchExponential, [0], 0.0)
        self.assertRaises(Exception, TorchExponential, [0], -1.0)
        self.assertRaises(Exception, TorchExponential, [0], np.inf)
        self.assertRaises(Exception, TorchExponential, [0], np.nan)

        # ----- support -----
        l = 1.5

        exponential = TorchExponential([0], l)

        # create test inputs/outputs
        data = torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]])# TODO (fails):, [0.0]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        # TODO (fails): self.assertTrue(all(probs[1] != 0.0))

    def test_gamma(self):
        torch.set_default_dtype(torch.float64)

        # ----- check inference -----

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = TorchGamma([0], alpha, beta)
        node_gamma = Gamma([0], alpha, beta)

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 1)

        log_probs = log_likelihood(node_gamma, data)
        log_probs_torch = log_likelihood(torch_gamma, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

        # ----- check gradient computation -----

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_gamma.alpha.grad is not None)
        self.assertTrue(torch_gamma.beta.grad is not None)

        alpha_orig = torch_gamma.alpha.detach().clone()
        beta_orig = torch_gamma.beta.detach().clone()

        optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(alpha_orig - torch_gamma.alpha.grad, torch_gamma.alpha))
        self.assertTrue(torch.allclose(beta_orig - torch_gamma.beta.grad, torch_gamma.beta))

        # verify that distribution paramters are also correctly updated (match parameters)
        self.assertTrue(torch.allclose(torch_gamma.alpha, torch_gamma.dist.concentration))
        self.assertTrue(torch.allclose(torch_gamma.beta, torch_gamma.dist.rate))

        # reset torch distribution after gradient update
        torch_gamma = TorchGamma([0], alpha, beta)

        # ----- check conversion between python and backend -----

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gamma.get_params()]),
                np.array([*toNodes(torch_gamma).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gamma.get_params()]), np.array([*toTorch(node_gamma).get_params()])
            )
        )

         # ----- (invalid) parameters -----
        TorchGamma([0], torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)), 1.0)
        TorchGamma([0], 1.0, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
        self.assertRaises(Exception, TorchGamma, [0], np.nextafter(0.0, -1.0), 1.0)
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, TorchGamma, [0], np.inf, 1.0)
        self.assertRaises(Exception, TorchGamma, [0], np.nan, 1.0)
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.inf)
        self.assertRaises(Exception, TorchGamma, [0], 1.0, np.nan)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
