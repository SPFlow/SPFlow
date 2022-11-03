from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.torch.structure.spn.nodes.sum_node import SPNSumNode
from spflow.torch.structure.spn.nodes.product_node import SPNProductNode
from spflow.torch.inference.spn.nodes.sum_node import log_likelihood
from spflow.torch.inference.spn.nodes.product_node import log_likelihood
from spflow.torch.learning.spn.nodes.sum_node import em
from spflow.torch.learning.spn.nodes.product_node import em
from spflow.torch.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial,
)
from spflow.torch.learning.nodes.leaves.parametric.negative_binomial import (
    maximum_likelihood_estimation,
    em,
)
from spflow.torch.inference.nodes.leaves.parametric.negative_binomial import (
    log_likelihood,
)
from spflow.torch.learning.expectation_maximization.expectation_maximization import (
    expectation_maximization,
)

import torch
import numpy as np
import unittest
import random


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

        leaf = NegativeBinomial(Scope([0]), n=3)

        # simulate data
        data = np.random.negative_binomial(n=3, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(
            torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-3)
        )

    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = NegativeBinomial(Scope([0]), n=10)

        # simulate data
        data = np.random.negative_binomial(n=10, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(
            torch.isclose(leaf.p, torch.tensor(0.7), atol=1e-2, rtol=1e-3)
        )

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = NegativeBinomial(Scope([0]), n=3)

        # simulate data
        data = np.random.negative_binomial(n=3, p=0.0, size=(100, 1))
        data[data < 0] = np.iinfo(
            data.dtype
        ).max  # p=zero leads to integer overflow due to infinite number of trials

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = NegativeBinomial(Scope([0]), n=5)

        # simulate data
        data = np.random.negative_binomial(n=5, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p < 1.0)

    def test_mle_only_nans(self):

        leaf = NegativeBinomial(Scope([0]), n=3)

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

        leaf = NegativeBinomial(Scope([0]), n=3)

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
            torch.tensor([[-0.1]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [2], [1]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        maximum_likelihood_estimation(
            leaf,
            torch.tensor([[float("nan")], [1], [2], [1]]),
            nan_strategy="ignore",
        )
        self.assertTrue(
            torch.isclose(
                leaf.p, torch.tensor((3 * 2) / (1 + 2 + 2 + 2 + 1 + 2))
            )
        )

    def test_mle_nan_strategy_callable(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        # should not raise an issue
        maximum_likelihood_estimation(
            leaf, torch.tensor([[1], [0], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [1]]),
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

        leaf = NegativeBinomial(Scope([0]), n=3)

        data = torch.tensor(
            np.vstack(
                [
                    np.random.negative_binomial(n=3, p=0.8, size=(10000, 1)),
                    np.random.negative_binomial(n=3, p=0.2, size=(10000, 1)),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.isclose(leaf.p, torch.tensor(0.2), atol=1e-3, rtol=1e-2)
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = NegativeBinomial(Scope([0]), n=3)
        data = torch.tensor(
            np.random.negative_binomial(n=3, p=0.3, size=(10000, 1))
        )
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(
            torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-3)
        )

    def test_em_product_of_negative_binomials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = NegativeBinomial(Scope([0]), n=3, p=0.5)
        l2 = NegativeBinomial(Scope([1]), n=5, p=0.5)
        prod_node = SPNProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.negative_binomial(n=3, p=0.8, size=(10000, 1)),
                    np.random.negative_binomial(n=5, p=0.2, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.isclose(l1.p, torch.tensor(0.8), atol=1e-3, rtol=1e-2)
        )
        self.assertTrue(
            torch.isclose(l2.p, torch.tensor(0.2), atol=1e-3, rtol=1e-2)
        )

    def test_em_sum_of_negative_binomials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = NegativeBinomial(Scope([0]), n=3, p=0.4)
        l2 = NegativeBinomial(Scope([0]), n=3, p=0.6)
        sum_node = SPNSumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.negative_binomial(n=3, p=0.8, size=(10000, 1)),
                    np.random.negative_binomial(n=3, p=0.3, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)
        self.assertTrue(
            torch.isclose(l1.p, torch.tensor(0.3), atol=1e-2, rtol=1e-2)
        )
        self.assertTrue(
            torch.isclose(l2.p, torch.tensor(0.8), atol=1e-2, rtol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
