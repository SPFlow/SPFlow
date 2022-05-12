from spflow.base.structure.nodes.leaves.parametric import Binomial
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchBinomial, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood
from spflow.torch.sampling import sample

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

        log_probs = log_likelihood(node_binomial, data, SPN())
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
        data = torch.distributions.Binomial(5, p_target).sample((100000, 1))

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

        self.assertTrue(
            torch.allclose(torch_binomial.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3)
        )

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

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        # p = 0
        binomial = TorchBinomial([0], 1, 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

        # p = 1
        binomial = TorchBinomial([0], 1, 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(
            Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
        )
        self.assertRaises(
            Exception, TorchBinomial, [0], 1, torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))
        )

        # p = inf and p = nan
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, TorchBinomial, [0], 1, np.nan)

        # n = 0
        binomial = TorchBinomial([0], 0, 0.5)

        data = torch.tensor([[0.0]])
        targets = torch.tensor([[1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, TorchBinomial, [0], -1, 0.5)

        # n float
        self.assertRaises(Exception, TorchBinomial, [0], 0.5, 0.5)

        # n = inf and n = nan
        self.assertRaises(Exception, TorchBinomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0], np.nan, 0.5)

        # invalid scope lengths
        self.assertRaises(Exception, TorchBinomial, [], 1, 0.5)
        self.assertRaises(Exception, TorchBinomial, [0, 1], 1, 0.5)

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        binomial = TorchBinomial([0], 2, 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[np.inf]]))

        # check valid integers inside valid range
        log_likelihood(binomial, torch.unsqueeze(torch.FloatTensor(list(range(binomial.n + 1))), 1))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-1]]))
        self.assertRaises(
            ValueError, log_likelihood, binomial, torch.tensor([[float(binomial.n + 1)]])
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(binomial.n)), torch.tensor(float(binomial.n + 1))
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[torch.nextafter(torch.tensor(float(binomial.n)), torch.tensor(0.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[0.5]]))
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[3.5]]))

    def test_marginalization(self):

        binomial = TorchBinomial([0], 5, 0.5)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_sampling(self):

        # ----- p = 0 -----

        binomial = TorchBinomial([0], 10, 0.0)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(binomial, data, ll_cache={}, instance_ids=[0, 2])

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~samples.isnan()] == 0.0))

        # ----- p = 1 -----

        binomial = TorchBinomial([0], 10, 1.0)

        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(binomial, data, ll_cache={}, instance_ids=[0, 2])

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))
        self.assertTrue(all(samples[~samples.isnan()] == 10))

        # ----- p = 0.5 -----

        binomial = TorchBinomial([0], 10, 0.5)

        samples = sample(binomial, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(5.0), rtol=0.01))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
