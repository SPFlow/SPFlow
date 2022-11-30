import random
import unittest

import numpy as np

from spflow.base.learning import maximum_likelihood_estimation
from spflow.base.structure.spn import Bernoulli
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.isclose(leaf.p, np.array(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.isclose(leaf.p, np.array(0.7), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=0.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

        # simulate data
        data = np.random.binomial(n=1, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p < 1.0)

    def test_mle_only_nans(self):

        leaf = Bernoulli(Scope([0]))

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

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = Bernoulli(Scope([0]))

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
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[2]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = Bernoulli(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [1]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = Bernoulli(Scope([0]))
        maximum_likelihood_estimation(
            leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy="ignore"
        )
        self.assertTrue(np.isclose(leaf.p, 2.0 / 3.0))

    def test_mle_nan_strategy_callable(self):

        leaf = Bernoulli(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(
            leaf, np.array([[1], [0], [1]]), nan_strategy=lambda x: x
        )

    def test_mle_nan_strategy_invalid(self):

        leaf = Bernoulli(Scope([0]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [1]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.nan], [1], [0], [1]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = Bernoulli(Scope([0]))

        data = np.vstack(
            [
                np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                np.random.binomial(n=1, p=0.2, size=(10000, 1)),
            ]
        )
        weights = np.concatenate([np.zeros(10000), np.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.isclose(leaf.p, 0.2, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
