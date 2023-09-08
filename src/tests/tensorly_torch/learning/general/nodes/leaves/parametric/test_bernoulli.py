import random
import unittest

import numpy as np
import pytest
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.torch.structure.spn import Bernoulli#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import updateBackend

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

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.7), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p < 1.0)

    def test_mle_only_nans(self):

        leaf = Bernoulli(Scope([0]))

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

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

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
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[2]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Bernoulli(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            torch.tensor([[float("nan")], [1], [0], [1]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Bernoulli(Scope([0]))
        maximum_likelihood_estimation(
            leaf,
            torch.tensor([[float("nan")], [1], [0], [1]]),
            nan_strategy="ignore",
        )
        self.assertTrue(torch.isclose(leaf.p, torch.tensor(2.0 / 3.0)))

    def test_mle_nan_strategy_callable(self):

        leaf = Bernoulli(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[1], [0], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Bernoulli(Scope([0]))
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

        leaf = Bernoulli(Scope([0]))

        data = torch.tensor(
            np.vstack(
                [
                    np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                    np.random.binomial(n=1, p=0.2, size=(10000, 1)),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.2), atol=1e-3, rtol=1e-2))

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))
        data = torch.tensor(np.random.binomial(n=1, p=0.2, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.2), atol=1e-2, rtol=1e-3))

    def test_em_product_of_bernoullis(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Bernoulli(Scope([0]), p=0.5)
        l2 = Bernoulli(Scope([1]), p=0.5)
        prod_node = ProductNode([l1, l2])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                    np.random.binomial(n=1, p=0.2, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.isclose(l1.p, torch.tensor(0.8), atol=1e-3, rtol=1e-2))
        self.assertTrue(torch.isclose(l2.p, torch.tensor(0.2), atol=1e-3, rtol=1e-2))

    def test_em_sum_of_bernoullis(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = Bernoulli(Scope([0]), p=0.4)
        l2 = Bernoulli(Scope([0]), p=0.6)
        sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.binomial(n=1, p=0.8, size=(1000, 1)),
                    np.random.binomial(n=1, p=0.3, size=(1000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        # optimal p
        p_opt = data.sum() / data.shape[0]
        # total p represented by mixture
        p_em = (sum_node.weights * torch.tensor([l1.p, l2.p])).sum()

        self.assertTrue(torch.isclose(p_opt, p_em))

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, tl.tensor(data))

        params = leaf.get_params()[0]

        leaf = Bernoulli(Scope([0]))
        prod_node = ProductNode([leaf])
        expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
        params_em = leaf.get_params()[0]


        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            leaf = Bernoulli(Scope([0]))
            leaf_updated = updateBackend(leaf)
            maximum_likelihood_estimation(leaf_updated, tl.tensor(data))
            # check conversion from torch to python
            self.assertTrue(np.isclose(leaf_updated.get_params()[0], params, atol=1e-2, rtol=1e-3))

            leaf = Bernoulli(Scope([0]))
            leaf_updated = updateBackend(leaf)
            prod_node = ProductNode([leaf_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                self.assertTrue(np.isclose(leaf_updated.get_params()[0], params_em, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
