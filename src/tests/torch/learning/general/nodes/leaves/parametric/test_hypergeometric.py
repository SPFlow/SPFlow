import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.torch.structure.spn import Hypergeometric, ProductNode, SumNode


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_mle(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # simulate data
        data = np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(10000, 1))

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(torch.tensor([leaf.N, leaf.M, leaf.n]) == torch.tensor([10, 7, 3])))

    def test_mle_invalid_support(self):

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("inf")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[-1]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[4]]),
            bias_correction=True,
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)
        data = torch.tensor(np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.requires_grad = True
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.all(torch.tensor([leaf.N, leaf.M, leaf.n]) == torch.tensor([10, 7, 3])))

    def test_em_product_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Hypergeometric(Scope([0]), N=10, M=3, n=5)
        l2 = Hypergeometric(Scope([1]), N=6, M=4, n=2)
        prod_node = ProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)),
                    np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(15000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.all(torch.tensor([l1.N, l1.M, l1.n]) == torch.tensor([10, 3, 5])))
        self.assertTrue(torch.all(torch.tensor([l2.N, l2.M, l2.n]) == torch.tensor([6, 4, 2])))

    def test_em_sum_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Hypergeometric(Scope([0]), N=10, M=3, n=5)
        l2 = Hypergeometric(Scope([0]), N=10, M=3, n=5)
        sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)))

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(torch.all(torch.tensor([l1.N, l1.M, l1.n]) == torch.tensor([10, 3, 5])))
        self.assertTrue(torch.all(torch.tensor([l2.N, l2.M, l2.n]) == torch.tensor([10, 3, 5])))


if __name__ == "__main__":
    unittest.main()
