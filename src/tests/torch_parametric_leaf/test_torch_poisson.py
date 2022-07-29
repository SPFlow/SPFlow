from spflow.base import sampling
from spflow.base.sampling.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric import Poisson
from spflow.base.inference import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric import TorchPoisson, toNodes, toTorch
from spflow.torch.inference import log_likelihood, likelihood
from spflow.torch.sampling import sample

from spflow.base.structure.network_type import SPN

import torch
import numpy as np

import random
import unittest


class TestTorchPoisson(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs = log_likelihood(node_poisson, data, SPN())
        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 10, (3, 1))

        log_probs_torch = log_likelihood(torch_poisson, torch.tensor(data))

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_poisson.l_aux.grad is not None)

        l_aux_orig = torch_poisson.l_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux))

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_poisson.l, torch_poisson.dist.rate))

    def test_gradient_optimization(self):

        # initialize distribution
        torch_poisson = TorchPoisson([0], l=1.0)

        torch.manual_seed(0)

        # create dummy data
        data = torch.distributions.Poisson(rate=4.0).sample((100000, 1))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=0.1)

        # perform optimization (possibly overfitting)
        for i in range(40):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_poisson, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(torch.allclose(torch_poisson.l, torch.tensor(4.0), atol=1e-3, rtol=0.3))

    def test_base_backend_conversion(self):

        l = random.randint(1, 10)

        torch_poisson = TorchPoisson([0], l)
        node_poisson = Poisson([0], l)

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

    def test_initialization(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)

        # l = 0
        TorchPoisson([0], 0.0)
        # l > 0
        TorchPoisson([0], torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))

        # l = -inf and l = inf
        self.assertRaises(Exception, TorchPoisson, [0], -np.inf)
        self.assertRaises(Exception, TorchPoisson, [0], np.inf)
        # l = nan
        self.assertRaises(Exception, TorchPoisson, [0], np.nan)

        # invalid scope lengths
        self.assertRaises(Exception, TorchPoisson, [], 1)
        self.assertRaises(Exception, TorchPoisson, [0, 1], 1)

    def test_support(self):

        # Support for Poisson distribution: integers N U {0}

        l = random.random()

        poisson = TorchPoisson([0], l)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, poisson, torch.tensor([[-float("inf")]]))
        self.assertRaises(ValueError, log_likelihood, poisson, torch.tensor([[float("inf")]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, poisson, torch.tensor([[-1]]))

        # check valid integers within valid range
        log_likelihood(poisson, torch.tensor([[0]]))
        log_likelihood(poisson, torch.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            torch.tensor([[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]),
        )
        self.assertRaises(ValueError, log_likelihood, poisson, torch.tensor([[10.1]]))

    def test_marginalization(self):

        poisson = TorchPoisson([0], 1.0)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(poisson, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_sampling(self):

        # ----- l = 1.0 -----

        poisson = TorchPoisson([0], 1.0)
        data = torch.tensor([[float("nan")], [float("nan")], [float("nan")]])

        samples = sample(poisson, data, ll_cache={}, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(samples.isnan() == torch.tensor([[False], [True], [False]])))

        samples = sample(poisson, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(1.0), rtol=0.1))

        # ----- l = 0.5 -----

        poisson = TorchPoisson([0], 0.5)

        samples = sample(poisson, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(0.5), rtol=0.1))

        # ----- l = 2.5 -----

        poisson = TorchPoisson([0], 2.5)

        samples = sample(poisson, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(2.5), rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
