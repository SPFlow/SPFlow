from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.learning.nodes.node import em
from spflow.torch.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.torch.learning.nodes.leaves.parametric.poisson import (
    maximum_likelihood_estimation,
    em,
)
from spflow.torch.inference.nodes.leaves.parametric.poisson import (
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

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=0.3, size=(10000, 1))

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

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=2.7, size=(50000, 1))

        # perform MLE
        maximum_likelihood_estimation(
            leaf, torch.tensor(data), bias_correction=True
        )

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(2.7), atol=1e-2, rtol=1e-2)
        )

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=1.0, size=(1, 1))

        # perform MLE
        maximum_likelihood_estimation(
            leaf, torch.tensor(data), bias_correction=True
        )

        self.assertFalse(torch.isnan(leaf.l))
        self.assertTrue(leaf.l > 0.0)

    def test_mle_only_nans(self):

        leaf = Poisson(Scope([0]))

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

        leaf = Poisson(Scope([0]))

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

        leaf = Poisson(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [2]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Poisson(Scope([0]))
        maximum_likelihood_estimation(
            leaf,
            torch.tensor([[float("nan")], [1], [0], [2]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        l_ignore = leaf.l

        maximum_likelihood_estimation(
            leaf,
            torch.tensor([[1], [0], [2]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        l_none = leaf.l

        self.assertTrue(torch.isclose(l_ignore, l_none))

    def test_mle_nan_strategy_callable(self):

        leaf = Poisson(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            leaf, torch.tensor([[2], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        leaf = Poisson(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [2]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [2]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = Poisson(Scope([0]))

        data = torch.tensor(
            np.vstack(
                [
                    np.random.poisson(1.7, size=(10000, 1)),
                    np.random.poisson(0.5, size=(10000, 1)),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.isclose(leaf.l, torch.tensor(0.5), atol=1e-2, rtol=1e-1)
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Poisson(Scope([0]))
        data = torch.tensor(np.random.poisson(lam=0.3, size=(10000, 1)))
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

    def test_em_product_of_poissons(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Poisson(Scope([0]))
        l2 = Poisson(Scope([1]))
        prod_node = SPNProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.poisson(lam=0.8, size=(15000, 1)),
                    np.random.poisson(lam=1.4, size=(15000, 1)),
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

    def test_em_sum_of_poissons(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Poisson(Scope([0]))
        l2 = Poisson(Scope([0]))
        sum_node = SPNSumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.poisson(lam=0.8, size=(10000, 1)),
                    np.random.poisson(lam=1.4, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        # optimal l
        l_opt = data.sum() / data.shape[0]
        # total l represented by mixture
        l_em = (sum_node.weights * torch.tensor([l1.l, l2.l])).sum()

        self.assertTrue(torch.isclose(l_opt, l_em))


if __name__ == "__main__":
    unittest.main()
