from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.learning.nodes.node import em
from spflow.torch.structure.layers.leaves.parametric.exponential import (
    ExponentialLayer,
)
from spflow.torch.learning.layers.leaves.parametric.exponential import (
    maximum_likelihood_estimation,
    em,
)
from spflow.torch.inference.layers.leaves.parametric.exponential import (
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

    def test_mle(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.exponential(scale=1.0 / 0.3, size=(20000, 1)),
                np.random.exponential(scale=1.0 / 2.7, size=(20000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(
            layer, torch.tensor(data), bias_correction=True
        )

        self.assertTrue(
            torch.allclose(
                layer.l, torch.tensor([0.3, 2.7]), atol=1e-2, rtol=1e-3
            )
        )

    def test_mle_bias_correction(self):

        layer = ExponentialLayer(Scope([0]))
        data = torch.tensor([[0.3], [2.7]])

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=False)
        self.assertTrue(torch.isclose(layer.l, torch.tensor(2.0 / 3.0)))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=True)
        self.assertTrue(torch.isclose(layer.l, torch.tensor(1.0 / 3.0)))

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0, size=(1, 1))

        # perform MLE (bias correction leads to zero result)
        maximum_likelihood_estimation(
            layer, torch.tensor(data), bias_correction=True
        )

        self.assertFalse(torch.isnan(layer.l))
        self.assertTrue(torch.all(layer.l > 0.0))

    def test_mle_only_nans(self):

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), 0.5]])

        # check if exception is raised
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            data,
            nan_strategy="ignore",
        )

    def test_mle_invalid_support(self):

        layer = ExponentialLayer(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[-0.1]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        layer = ExponentialLayer(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        layer = ExponentialLayer(Scope([0]))
        maximum_likelihood_estimation(
            layer,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        self.assertTrue(torch.isclose(layer.l, torch.tensor(3.0 / 2.7)))

    def test_mle_nan_strategy_callable(self):

        layer = ExponentialLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            layer, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        layer = ExponentialLayer(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [1], [0], [1]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = ExponentialLayer([Scope([0]), Scope([1])])

        data = torch.tensor(
            np.hstack(
                [
                    np.vstack(
                        [
                            np.random.exponential(
                                scale=1.0 / 1.8, size=(10000, 1)
                            ),
                            np.random.exponential(
                                scale=1.0 / 0.2, size=(10000, 1)
                            ),
                        ]
                    ),
                    np.vstack(
                        [
                            np.random.exponential(
                                scale=1.0 / 0.3, size=(10000, 1)
                            ),
                            np.random.exponential(
                                scale=1.0 / 1.7, size=(10000, 1)
                            ),
                        ]
                    ),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.allclose(
                leaf.l, torch.tensor([0.2, 1.7]), atol=1e-3, rtol=1e-2
            )
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer([Scope([0]), Scope([1])])
        data = torch.tensor(
            np.hstack(
                [
                    np.random.exponential(scale=1.0 / 0.3, size=(10000, 1)),
                    np.random.exponential(scale=1.0 / 1.7, size=(10000, 1)),
                ]
            )
        )
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(layer, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(
            torch.allclose(
                layer.l, torch.tensor([0.3, 1.7]), atol=1e-2, rtol=1e-2
            )
        )

    def test_em_product_of_exponentials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer([Scope([0]), Scope([1])])
        prod_node = SPNProductNode([layer])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.exponential(scale=1.0 / 0.8, size=(10000, 1)),
                    np.random.exponential(scale=1.0 / 1.4, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                layer.l, torch.tensor([0.8, 1.4]), atol=1e-3, rtol=1e-2
            )
        )

    def test_em_sum_of_exponentials(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer([Scope([0]), Scope([0])], l=[0.6, 1.2])
        sum_node = SPNSumNode([layer], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.exponential(scale=1.0 / 0.8, size=(10000, 1)),
                    np.random.exponential(scale=1.0 / 1.4, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                layer.l, torch.tensor([0.8, 1.4]), atol=1e-2, rtol=1e-2
            )
        )


if __name__ == "__main__":
    unittest.main()
