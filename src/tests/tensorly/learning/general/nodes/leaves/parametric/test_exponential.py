import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isclose, tl_vstack, tl_isnan

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import Exponential
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0 / 0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(tl_isclose(leaf.l, tl.tensor(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0 / 2.7, size=(50000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(tl_isclose(leaf.l, tl.tensor(2.7), atol=1e-2, rtol=1e-2))

    def test_mle_bias_correction(self):

        leaf = Exponential(Scope([0]))
        data = tl.tensor([[0.3], [2.7]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(tl_isclose(leaf.l, 2.0 / 3.0))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(tl_isclose(leaf.l, 1.0 / 3.0))

    def test_mle_edge_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Exponential(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0, size=(1, 1))

        # perform MLE (bias correction leads to zero result)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertFalse(tl_isnan(leaf.l))
        self.assertTrue(leaf.l > 0.0)

    def test_mle_only_nans(self):

        leaf = Exponential(Scope([0]))

        # simulate data
        data = tl.tensor([[tl.nan], [tl.nan]])

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
            tl.tensor([[tl.inf]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[-0.1]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Exponential(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan], [0.1], [1.9], [0.7]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Exponential(Scope([0]))
        maximum_likelihood_estimation(
            leaf,
            tl.tensor([[tl.nan], [0.1], [1.9], [0.7]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        self.assertTrue(tl_isclose(leaf.l, 3.0 / 2.7))

    def test_mle_nan_strategy_callable(self):

        leaf = Exponential(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, tl.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Exponential(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan], [0.1], [1.9], [0.7]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan], [1], [0], [1]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = Exponential(Scope([0]))

        data = np.vstack(
            [
                np.random.exponential(1.0 / 0.8, size=(10000, 1)),
                np.random.exponential(1.0 / 1.4, size=(10000, 1)),
            ]
        )
        weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(tl_isclose(leaf.l, 1.4, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
