from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import SumNode, ProductNode, Exponential
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)

import torch
import numpy as np
import random
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_mle_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0 / 0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(
            leaf, torch.tensor(data), bias_correction=True
        )

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(0.3), atol=1e-2, rtol=1e-3)
        )

    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0 / 2.7, size=(50000, 1))

        # perform MLE
        maximum_likelihood_estimation(
            leaf, torch.tensor(data), bias_correction=True
        )

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(2.7), atol=1e-2, rtol=1e-2)
        )

    def test_mle_bias_correction(self):

        leaf = Exponential(Scope([0]))
        data = torch.tensor([[0.3], [2.7]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(torch.isclose(leaf.l, torch.tensor(2.0 / 3.0)))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(torch.isclose(leaf.l, torch.tensor(1.0 / 3.0)))

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0, size=(1, 1))

        # perform MLE (bias correction leads to zero result)
        maximum_likelihood_estimation(
            leaf, torch.tensor(data), bias_correction=True
        )

        self.assertFalse(torch.isnan(leaf.l))
        self.assertTrue(leaf.l > 0.0)

    def test_mle_only_nans(self):

        leaf = Exponential(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan")], [float("nan")]])

        # check if exception is raised
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            data,
            nan_strategy="ignore",
        )

    def test_mle_invalid_support(self):

        leaf = Exponential(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[-0.1]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Exponential(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Exponential(Scope([0]))
        maximum_likelihood_estimation(
            leaf,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        self.assertTrue(torch.isclose(leaf.l, torch.tensor(3.0 / 2.7)))

    def test_mle_nan_strategy_callable(self):

        leaf = Exponential(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            leaf, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        leaf = Exponential(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [1]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = Exponential(Scope([0]))

        data = torch.tensor(
            np.vstack(
                [
                    np.random.exponential(1.0 / 0.8, size=(10000, 1)),
                    np.random.exponential(1.0 / 1.4, size=(10000, 1)),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(1.4), atol=1e-3, rtol=1e-2)
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))
        data = torch.tensor(np.random.exponential(1.0 / 0.3, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(0.3), atol=1e-2, rtol=1e-3)
        )

    def test_em_product_of_exponentials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Exponential(Scope([0]))
        l2 = Exponential(Scope([1]))
        prod_node = ProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.exponential(1.0 / 0.8, size=(10000, 1)),
                    np.random.exponential(1.0 / 1.4, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.isclose(l1.l, torch.tensor(0.8), atol=1e-3, rtol=1e-2)
        )
        self.assertTrue(
            torch.isclose(l2.l, torch.tensor(1.4), atol=1e-3, rtol=1e-2)
        )

    def test_em_sum_of_exponentials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Exponential(Scope([0]), l=0.6)
        l2 = Exponential(Scope([0]), l=1.2)
        sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.exponential(1.0 / 0.8, size=(10000, 1)),
                    np.random.exponential(1.0 / 1.4, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(
            torch.isclose(l1.l, torch.tensor(0.8), atol=1e-2, rtol=1e-2)
        )
        self.assertTrue(
            torch.isclose(l2.l, torch.tensor(1.4), atol=1e-2, rtol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
