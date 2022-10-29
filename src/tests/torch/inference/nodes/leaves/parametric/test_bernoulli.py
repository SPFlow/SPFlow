from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.bernoulli import (
    Bernoulli as BaseBernoulli,
)
from spflow.base.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.bernoulli import (
    Bernoulli,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.bernoulli import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestBernoulli(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        p = random.random()

        torch_bernoulli = Bernoulli(Scope([0]), p)
        node_bernoulli = BaseBernoulli(Scope([0]), p)

        # create dummy input data (batch size x random variables)
        data = np.random.randint(0, 2, (3, 1))

        log_probs = log_likelihood(node_bernoulli, data)
        log_probs_torch = log_likelihood(torch_bernoulli, torch.tensor(data))

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        p = random.random()

        torch_bernoulli = Bernoulli(Scope([0]), p)

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
            torch.allclose(
                p_aux_orig - torch_bernoulli.p_aux.grad, torch_bernoulli.p_aux
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(torch_bernoulli.p, torch_bernoulli.dist.probs)
        )

    def test_gradient_optimization(self):

        torch.manual_seed(0)

        # initialize distribution
        torch_bernoulli = Bernoulli(Scope([0]), 0.3)

        # create dummy data
        p_target = 0.8
        data = torch.bernoulli(torch.full((100000, 1), p_target))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(
            torch_bernoulli.parameters(), lr=0.5, momentum=0.5
        )

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
            torch.allclose(
                torch_bernoulli.p, torch.tensor(p_target), atol=1e-3, rtol=1e-3
            )
        )

    def test_likelihood_p_0(self):

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)

        data = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))
        self.assertTrue(torch.allclose(probs, targets))

    def test_likelihood_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), random.random())
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(bernoulli, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        p = random.random()
        bernoulli = Bernoulli(Scope([0]), p)

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[-float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor([[float("inf")]]),
        )

        # check valid integers inside valid range
        log_likelihood(bernoulli, torch.tensor([[0.0], [1.0]]))

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, torch.tensor([[-1]])
        )
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, torch.tensor([[2]])
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor(
                [[torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor(
                [[torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            torch.tensor(
                [[torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))]]
            ),
        )
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, torch.tensor([[0.5]])
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
