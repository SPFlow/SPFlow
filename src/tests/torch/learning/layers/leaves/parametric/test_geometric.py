from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.torch.structure.spn.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.spn.nodes.node import log_likelihood
from spflow.torch.learning.spn.nodes.node import em
from spflow.torch.structure.layers.leaves.parametric.geometric import (
    GeometricLayer,
)
from spflow.torch.learning.layers.leaves.parametric.geometric import (
    maximum_likelihood_estimation,
    em,
)
from spflow.torch.inference.layers.leaves.parametric.geometric import (
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

        layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.geometric(p=0.3, size=(10000, 1)),
                np.random.geometric(p=0.7, size=(10000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(
            layer, torch.tensor(data), bias_correction=True
        )
        self.assertTrue(
            torch.allclose(
                layer.p, torch.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-2
            )
        )

    def test_mle_bias_correction(self):

        layer = GeometricLayer(Scope([0]))
        data = torch.tensor([[1.0], [3.0]])

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=False)
        self.assertTrue(torch.allclose(layer.p, torch.tensor(2.0 / 4.0)))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=True)
        self.assertTrue(torch.allclose(layer.p, torch.tensor(1.0 / 4.0)))

    def test_mle_edge_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = GeometricLayer(Scope([0]))

        # simulate data
        data = torch.ones(100, 1)

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=True)
        self.assertTrue(torch.all(layer.p < 1.0))

    def test_mle_only_nans(self):

        layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

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

        layer = GeometricLayer(Scope([0]))

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
            torch.tensor([[0]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        layer = GeometricLayer(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [1], [4], [3]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        layer = GeometricLayer(Scope([0]))
        maximum_likelihood_estimation(
            layer,
            torch.tensor([[float("nan")], [1], [4], [3]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        self.assertTrue(torch.allclose(layer.p, torch.tensor(3.0 / 8.0)))

    def test_mle_nan_strategy_callable(self):

        layer = GeometricLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            layer, torch.tensor([[2], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        layer = GeometricLayer(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [1], [4], [3]]),
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

        leaf = GeometricLayer([Scope([0]), Scope([1])], n_nodes=3)

        data = torch.tensor(
            np.hstack(
                [
                    np.vstack(
                        [
                            np.random.geometric(p=0.8, size=(10000, 1)),
                            np.random.geometric(p=0.2, size=(10000, 1)),
                        ]
                    ),
                    np.vstack(
                        [
                            np.random.geometric(p=0.3, size=(10000, 1)),
                            np.random.geometric(p=0.7, size=(10000, 1)),
                        ]
                    ),
                ]
            )
        )

        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.allclose(
                leaf.p, torch.tensor([0.2, 0.7]), atol=1e-3, rtol=1e-2
            )
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = GeometricLayer([Scope([0]), Scope([1])])
        data = torch.tensor(
            np.hstack(
                [
                    np.random.geometric(p=0.2, size=(10000, 1)),
                    np.random.geometric(p=0.7, size=(10000, 1)),
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
                layer.p, torch.tensor([0.2, 0.7]), atol=1e-2, rtol=1e-3
            )
        )

    def test_em_product_of_geometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = GeometricLayer([Scope([0]), Scope([1])])
        prod_node = SPNProductNode([layer])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.geometric(p=0.8, size=(10000, 1)),
                    np.random.geometric(p=0.2, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                layer.p, torch.tensor([0.8, 0.2]), atol=1e-3, rtol=1e-2
            )
        )

    def test_em_sum_of_geometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = GeometricLayer(Scope([0]), n_nodes=2, p=[0.4, 0.6])
        sum_node = SPNSumNode([leaf], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.geometric(p=0.8, size=(10000, 1)),
                    np.random.geometric(p=0.2, size=(10000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                leaf.p, torch.tensor([0.2, 0.8]), atol=1e-2, rtol=1e-2
            )
        )


if __name__ == "__main__":
    unittest.main()
