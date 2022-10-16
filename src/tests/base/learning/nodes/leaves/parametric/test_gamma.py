from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.learning.nodes.leaves.parametric.gamma import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = np.random.gamma(shape=0.3, scale=1.0/1.7, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.alpha, np.array(0.3), atol=1e-3, rtol=1e-2))
        self.assertTrue(np.isclose(leaf.beta, np.array(1.7), atol=1e-3, rtol=1e-2))
    
    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = np.random.gamma(shape=1.9, scale=1.0/0.7, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.alpha, np.array(1.9), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.isclose(leaf.beta, np.array(0.7), atol=1e-2, rtol=1e-2))

    def test_mle_only_nans(self):
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = np.array([[np.nan], [np.nan]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):

        leaf = Gamma(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.inf]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[-0.1]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = Gamma(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [0.1], [1.9], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = Gamma(Scope([0]))
        maximum_likelihood_estimation(leaf, np.array([[np.nan], [0.1], [1.9], [0.7]]), nan_strategy='ignore', bias_correction=False)
        alpha_ignore, beta_ignore = leaf.alpha, leaf.beta

        # since gamma is estimated iteratively by scipy, just make sure it matches the estimate without nan value
        maximum_likelihood_estimation(leaf, np.array([[0.1], [1.9], [0.7]]), nan_strategy=None, bias_correction=False)
        alpha_none, beta_none = leaf.alpha, leaf.beta

        self.assertTrue(np.isclose(alpha_ignore, alpha_none))
        self.assertTrue(np.isclose(beta_ignore, beta_none))

    def test_mle_nan_strategy_callable(self):

        leaf = Gamma(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, np.array([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Gamma(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy=1)

    def test_weighted_mle(self):

        leaf = Gamma(Scope([0]))

        data = np.vstack([
            np.random.gamma(shape=1.7, scale=1.0/0.8, size=(10000,1)),
            np.random.gamma(shape=0.5, scale=1.0/1.4, size=(10000,1))
        ])
        weights = np.concatenate([
            np.zeros(10000),
            np.ones(10000)
        ])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.isclose(leaf.alpha, 0.5, atol=1e-3, rtol=1e-2))
        self.assertTrue(np.isclose(leaf.beta, 1.4, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()