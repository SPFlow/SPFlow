import random
import unittest

import numpy as np

from spflow.base.learning import maximum_likelihood_estimation
from spflow.base.structure.spn import Poisson
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.l, np.array(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=2.7, size=(50000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.l, np.array(2.7), atol=1e-2, rtol=1e-2))

    def test_mle_edge_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=1.0, size=(1, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertFalse(np.isnan(leaf.l))
        self.assertTrue(leaf.l > 0.0)

    def test_mle_only_nans(self):

        leaf = Poisson(Scope([0]))

        # simulate data
        data = np.array([[np.nan], [np.nan]])

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
            np.array([[np.inf]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[-0.1]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Poisson(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [2]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Poisson(Scope([0]))
        maximum_likelihood_estimation(
            leaf,
            np.array([[np.nan], [1], [0], [2]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        l_ignore = leaf.l

        maximum_likelihood_estimation(
            leaf,
            np.array([[1], [0], [2]]),
            nan_strategy="ignore",
            bias_correction=False,
        )
        l_none = leaf.l

        self.assertTrue(np.isclose(l_ignore, l_none))

    def test_mle_nan_strategy_callable(self):

        leaf = Poisson(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            leaf, np.array([[2], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        leaf = Poisson(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [2]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [2]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = Poisson(Scope([0]))

        data = np.vstack(
            [
                np.random.poisson(1.7, size=(10000, 1)),
                np.random.poisson(0.5, size=(10000, 1)),
            ]
        )
        weights = np.concatenate([np.zeros(10000), np.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.isclose(leaf.l, 0.5, atol=1e-2, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
