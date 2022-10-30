from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric as BaseHypergeometric,
)
from spflow.base.inference.nodes.leaves.parametric.hypergeometric import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric,
    toBase,
    toTorch,
)
from spflow.torch.inference.nodes.leaves.parametric.hypergeometric import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestHypergeometric(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)
        node_hypergeometric = BaseHypergeometric(Scope([0]), N, M, n)

        # create dummy input data (batch size x random variables)
        data = np.array([[5], [10]])

        log_probs = log_likelihood(node_hypergeometric, data)
        log_probs_torch = log_likelihood(
            torch_hypergeometric, torch.tensor(data)
        )

        # TODO: support is handled differently (in log space): -inf for torch and np.finfo().min for numpy (decide how to handle)
        log_probs[log_probs == np.finfo(log_probs.dtype).min] = -np.inf

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # create dummy input data (batch size x random variables)
        data = np.array([[5], [10]])

        log_probs_torch = log_likelihood(
            torch_hypergeometric, torch.tensor(data)
        )

        # create dummy targets
        targets_torch = torch.ones(2, 1)
        targets_torch.requires_grad = True

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_hypergeometric.N.grad is None)
        self.assertTrue(torch_hypergeometric.M.grad is None)
        self.assertTrue(torch_hypergeometric.n.grad is None)

        # make sure distribution has no (learnable) parameters
        self.assertFalse(list(torch_hypergeometric.parameters()))

    def test_likelihood_marginalization(self):

        hypergeometric = Hypergeometric(Scope([0]), 15, 10, 10)
        data = torch.tensor([[float("nan")]])

        # should not raise and error and should return 1
        probs = likelihood(hypergeometric, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

    def test_support(self):

        # Support for Hypergeometric distribution: integers {max(0,n+M-N),...,min(n,M)}

        # case n+M-N > 0
        N = 15
        M = 10
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[-float("inf")]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[float("inf")]]),
        )

        # check valid integers inside valid range
        data = torch.tensor([[max(0, n + M - N)], [min(n, M)]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(all(probs != 0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[max(0, n + M - N) - 1]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[min(n, M) + 1]]),
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(max(0, n + M - N))),
                            torch.tensor(100),
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(max(n, M))), torch.tensor(-1.0)
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, torch.tensor([[5.5]])
        )

        # case n+M-N
        N = 25

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # check valid integers within valid range
        data = torch.tensor([[max(0, n + M - N)], [min(n, M)]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(all(probs != 0))
        self.assertTrue(torch.allclose(probs, torch.exp(log_probs)))

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[max(0, n + M - N) - 1]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor([[min(n, M) + 1]]),
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(max(0, n + M - N))),
                            torch.tensor(100),
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            torch.tensor(
                [
                    [
                        torch.nextafter(
                            torch.tensor(float(max(n, M))), torch.tensor(-1.0)
                        )
                    ]
                ]
            ),
        )
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, torch.tensor([[5.5]])
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
