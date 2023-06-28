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
from spflow.torch.structure.spn import HypergeometricLayer#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode


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

        layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])

        # simulate data
        data = np.hstack(
            [
                np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(1000, 1)),
                np.random.hypergeometric(ngood=2, nbad=4 - 2, nsample=2, size=(1000, 1)),
            ]
        )

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 4])))
        self.assertTrue(torch.all(layer.M == torch.tensor([7, 2])))
        self.assertTrue(torch.all(layer.n == torch.tensor([3, 2])))

    def test_mle_invalid_support(self):

        layer = HypergeometricLayer(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("inf")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[-1]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[4]]),
            bias_correction=True,
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
        data = torch.tensor(
            np.hstack(
                [
                    np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(10000, 1)),
                    np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(10000, 1)),
                ]
            )
        )

        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.requires_grad = True
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.all(leaf.N == torch.tensor([10, 6])))
        self.assertTrue(torch.all(leaf.M == torch.tensor([3, 4])))
        self.assertTrue(torch.all(leaf.n == torch.tensor([5, 2])))

    def test_em_product_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
        prod_node = ProductNode([layer])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)),
                    np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(15000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 6])))
        self.assertTrue(torch.all(layer.M == torch.tensor([3, 4])))
        self.assertTrue(torch.all(layer.n == torch.tensor([5, 2])))

    def test_em_sum_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = HypergeometricLayer(Scope([0]), n_nodes=2, N=10, M=3, n=5)
        sum_node = SumNode([layer], weights=[0.5, 0.5])

        data = torch.tensor(np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)))

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 10])))
        self.assertTrue(torch.all(layer.M == torch.tensor([3, 3])))
        self.assertTrue(torch.all(layer.n == torch.tensor([5, 5])))


if __name__ == "__main__":
    unittest.main()
