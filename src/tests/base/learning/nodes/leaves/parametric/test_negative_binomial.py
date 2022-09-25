from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import NegativeBinomial
from spflow.base.learning.nodes.leaves.parametric.negative_binomial import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = NegativeBinomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.isclose(leaf.p, np.array(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = NegativeBinomial(Scope([0]), n=10)

        # simulate data
        data = np.random.binomial(n=10, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(np.isclose(leaf.p, np.array(0.7), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = NegativeBinomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = NegativeBinomial(Scope([0]), n=5)

        # simulate data
        data = np.random.binomial(n=5, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data)

        self.assertTrue(leaf.p < 1.0)
    
    def test_mle_only_nans(self):
        
        leaf = NegativeBinomial(Scope([0]), n=3)

        # simulate data
        data = np.array([[np.nan], [np.nan]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):

        leaf = NegativeBinomial(Scope([0]), n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.inf]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[-0.1]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[4]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [2], [1]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        maximum_likelihood_estimation(leaf, np.array([[np.nan], [1], [2], [1]]), nan_strategy='ignore')
        self.assertTrue(np.isclose(leaf.p, 4.0/6.0))

    def test_mle_nan_strategy_callable(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        # should not raise an issue
        maximum_likelihood_estimation(leaf, np.array([[1], [0], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = NegativeBinomial(Scope([0]), n=2)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy=1)


if __name__ == "__main__":
    unittest.main()