import random
import unittest

import numpy as np
import torch

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Binomial as BaseBinomial
from spflow.meta.data import Scope
from spflow.torch.inference import likelihood, log_likelihood
#from spflow.torch.structure.spn import Binomial
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_binomial import Binomial


class TestBinomial(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = Binomial(Scope([0]), n, p)
        node_binomial = BaseBinomial(Scope([0]), n, p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(1, n, (3, 1))

        log_probs = log_likelihood(node_binomial, data)
        log_probs_torch = log_likelihood(torch_binomial, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(np.allclose(log_probs, log_probs_torch.detach().cpu().numpy()))

    def test_gradient_computation(self):

        n = random.randint(2, 10)
        p = random.random()

        torch_binomial = Binomial(Scope([0]), n, p)

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
        self.assertTrue(torch.allclose(p_aux_orig - torch_binomial.p_aux.grad, torch_binomial.p_aux))

        # verify that distribution parameters match parameters
        self.assertTrue(torch.equal(torch_binomial.n, torch_binomial.dist.total_count.long()))
        self.assertTrue(torch.allclose(torch_binomial.p, torch_binomial.dist.probs))

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_binomial = Binomial(Scope([0]), 5, 0.3)

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

        self.assertTrue(torch.allclose(torch_binomial.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3))

    def test_likelihood_p_0(self):

        # p = 0
        binomial = Binomial(Scope([0]), 1, 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        binomial = Binomial(Scope([0]), 1, 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_n_0(self):

        # n = 0
        binomial = Binomial(Scope([0]), 0, 0.5)

        data = torch.tensor([[0.0]])
        targets = torch.tensor([[1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_marginalization(self):

        binomial = Binomial(Scope([0]), 5, 0.5)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        binomial = Binomial(Scope([0]), 2, 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[np.inf]]))

        # check valid integers inside valid range
        log_likelihood(
            binomial,
            torch.unsqueeze(torch.FloatTensor(list(range(binomial.n + 1))), 1),
        )

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, torch.tensor([[-1]]))
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            torch.tensor([[float(binomial.n + 1)]]),
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
                            torch.tensor(float(binomial.n)),
                            torch.tensor(float(binomial.n + 1)),
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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
