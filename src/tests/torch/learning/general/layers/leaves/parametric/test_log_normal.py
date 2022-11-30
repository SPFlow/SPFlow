from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure.spn import SumNode, ProductNode, LogNormalLayer
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import (
    em,
    maximum_likelihood_estimation,
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

        layer = LogNormalLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.lognormal(mean=-1.7, sigma=0.2, size=(20000, 1)),
                np.random.lognormal(mean=0.5, sigma=1.3, size=(20000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(
            layer, torch.tensor(data), bias_correction=True
        )

        self.assertTrue(
            torch.allclose(
                layer.mean, torch.tensor([-1.7, 0.5]), atol=1e-2, rtol=1e-2
            )
        )
        self.assertTrue(
            torch.allclose(
                layer.std, torch.tensor([0.2, 1.3]), atol=1e-2, rtol=1e-2
            )
        )

    def test_mle_bias_correction(self):

        layer = LogNormalLayer(Scope([0]))
        data = torch.exp(torch.tensor([[-1.0], [1.0]]))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=False)
        self.assertTrue(torch.isclose(layer.std, torch.sqrt(torch.tensor(1.0))))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=True)
        self.assertTrue(torch.isclose(layer.std, torch.sqrt(torch.tensor(2.0))))

    def test_mle_edge_std_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer(Scope([0]))

        # simulate data
        data = torch.exp(torch.randn(1, 1))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=False)

        self.assertTrue(torch.allclose(layer.mean, torch.log(data[0])))
        self.assertTrue(layer.std > 0)

    def test_mle_edge_std_nan(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer(Scope([0]))

        # simulate data
        data = torch.exp(torch.randn(1, 1))

        # perform MLE (Torch does not throw a warning different to NumPy)
        maximum_likelihood_estimation(layer, data, bias_correction=True)

        self.assertTrue(torch.isclose(layer.mean, torch.log(data[0])))
        self.assertFalse(torch.isnan(layer.std))
        self.assertTrue(torch.all(layer.std > 0))

    def test_mle_only_nans(self):

        layer = LogNormalLayer(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), 2.0]])

        # check if exception is raised
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            data,
            nan_strategy="ignore",
        )

    def test_mle_invalid_support(self):

        layer = LogNormalLayer(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("inf")]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        layer = LogNormalLayer(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("nan")], [0.1], [-1.8], [0.7]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        layer = LogNormalLayer(Scope([0]))
        maximum_likelihood_estimation(
            layer,
            torch.exp(torch.tensor([[float("nan")], [0.1], [-1.8], [0.7]])),
            nan_strategy="ignore",
            bias_correction=False,
        )
        self.assertTrue(torch.allclose(layer.mean, torch.tensor(-1.0 / 3.0)))
        self.assertTrue(
            torch.allclose(
                layer.std,
                torch.sqrt(
                    1
                    / 3
                    * torch.sum(
                        (torch.tensor([[0.1], [-1.8], [0.7]]) + 1.0 / 3.0) ** 2
                    )
                ),
            )
        )

    def test_mle_nan_strategy_callable(self):

        layer = LogNormalLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            layer, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        layer = LogNormalLayer(Scope([0]))
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

        leaf = LogNormalLayer([Scope([0]), Scope([1])])

        data = torch.tensor(
            np.hstack(
                [
                    np.vstack(
                        [
                            np.random.lognormal(1.7, 0.8, size=(10000, 1)),
                            np.random.lognormal(0.5, 1.4, size=(10000, 1)),
                        ]
                    ),
                    np.vstack(
                        [
                            np.random.lognormal(0.9, 0.3, size=(10000, 1)),
                            np.random.lognormal(1.3, 1.7, size=(10000, 1)),
                        ]
                    ),
                ]
            )
        )
        weights = torch.concat([torch.zeros(10000), torch.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            torch.allclose(
                leaf.mean, torch.tensor([0.5, 1.3]), atol=1e-2, rtol=1e-1
            )
        )
        self.assertTrue(
            torch.allclose(
                leaf.std, torch.tensor([1.4, 1.7]), atol=1e-2, rtol=1e-1
            )
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer([Scope([0]), Scope([1])])
        data = torch.tensor(
            np.hstack(
                [
                    np.random.lognormal(0.3, 1.7, size=(10000, 1)),
                    np.random.lognormal(1.4, 0.8, size=(10000, 1)),
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
                layer.mean, torch.tensor([0.3, 1.4]), atol=1e-2, rtol=1e-1
            )
        )
        self.assertTrue(
            torch.allclose(
                layer.std, torch.tensor([1.7, 0.8]), atol=1e-2, rtol=1e-1
            )
        )

    def test_em_product_of_log_normals(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer(
            [Scope([0]), Scope([1])], mean=[1.5, -2.5], std=[0.75, 1.5]
        )
        prod_node = ProductNode([layer])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.lognormal(2.0, 1.0, size=(15000, 1)),
                    np.random.lognormal(-2.0, 1.0, size=(15000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                layer.mean, torch.tensor([2.0, -2.0]), atol=1e-2, rtol=1e-1
            )
        )
        self.assertTrue(
            torch.allclose(
                layer.std, torch.tensor([1.0, 1.0]), atol=1e-2, rtol=1e-1
            )
        )

    def test_em_sum_of_log_normals(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer(
            [Scope([0]), Scope([0])], mean=[1.5, -2.5], std=[0.75, 1.5]
        )
        sum_node = SumNode([layer], weights=[0.5, 0.5])

        data = torch.tensor(
            np.vstack(
                [
                    np.random.lognormal(2.0, 1.0, size=(20000, 1)),
                    np.random.lognormal(-2.0, 1.0, size=(20000, 1)),
                ]
            )
        )

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(
            torch.allclose(
                layer.mean, torch.tensor([2.0, -2.0]), atol=1e-2, rtol=1e-1
            )
        )
        self.assertTrue(
            torch.allclose(
                layer.std, torch.tensor([1.0, 1.0]), atol=1e-2, rtol=1e-1
            )
        )


if __name__ == "__main__":
    unittest.main()
