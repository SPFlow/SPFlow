import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isclose, tl_vstack

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import Binomial
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Binomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(tl_isclose(leaf.p, tl.tensor(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Binomial(Scope([0]), n=10)

        # simulate data
        data = np.random.binomial(n=10, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(tl_isclose(leaf.p, tl.tensor(0.7), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Binomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Binomial(Scope([0]), n=5)

        # simulate data
        data = np.random.binomial(n=5, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p < 1.0)

    def test_mle_only_nans(self):

        leaf = Binomial(Scope([0]), n=3)

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

        leaf = Binomial(Scope([0]), n=3)

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
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[4]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Binomial(Scope([0]), n=2)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan], [1], [2], [1]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Binomial(Scope([0]), n=2)
        maximum_likelihood_estimation(leaf, tl.tensor([[tl.nan], [1], [2], [1]]), nan_strategy="ignore")
        self.assertTrue(tl_isclose(leaf.p, 4.0 / 6.0))

    def test_mle_nan_strategy_callable(self):

        leaf = Binomial(Scope([0]), n=2)
        # should not raise an issue
        maximum_likelihood_estimation(leaf, tl.tensor([[1], [0], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Binomial(Scope([0]), n=2)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan], [1], [0], [1]]),
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

        leaf = Binomial(Scope([0]), n=3)

        data = np.vstack(
            [
                np.random.binomial(n=3, p=0.8, size=(10000, 1)),
                np.random.binomial(n=3, p=0.2, size=(10000, 1)),
            ]
        )
        weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(tl_isclose(leaf.p, 0.2, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
