from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.base.learning.nodes.leaves.parametric.log_normal import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.random.lognormal(mean=-1.7, sigma=0.2, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.mean, np.array(-1.7), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.isclose(leaf.std, np.array(0.2), atol=1e-2, rtol=1e-2))
    
    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.random.lognormal(mean=0.5, sigma=1.3, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.mean, np.array(0.5), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.isclose(leaf.std, np.array(1.3), atol=1e-2, rtol=1e-2))
    
    def test_mle_bias_correction(self):

        leaf = LogNormal(Scope([0]))
        data = np.exp(np.array([[-1.0], [1.0]]))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(np.isclose(leaf.std, np.sqrt(1.0)))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(np.isclose(leaf.std, np.sqrt(2.0)))

    def test_mle_edge_std_0(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.exp(np.random.randn(1, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)

        self.assertTrue(np.isclose(leaf.mean, np.log(data[0])))
        self.assertTrue(leaf.std > 0)

    def test_mle_edge_std_nan(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.exp(np.random.randn(1, 1))

        # perform MLE (should throw a warning due to bias correction on a single sample)
        self.assertWarns(RuntimeWarning, maximum_likelihood_estimation, leaf, data, bias_correction=True)

        self.assertTrue(np.isclose(leaf.mean, np.log(data[0])))
        self.assertFalse(np.isnan(leaf.std))
        self.assertTrue(leaf.std > 0)

    def test_mle_only_nans(self):
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.array([[np.nan], [np.nan]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):
        
        leaf = LogNormal(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.inf]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = LogNormal(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [0.1], [-1.8], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = LogNormal(Scope([0]))
        maximum_likelihood_estimation(leaf, np.exp(np.array([[np.nan], [0.1], [-1.8], [0.7]])), nan_strategy='ignore', bias_correction=False)
        self.assertTrue(np.isclose(leaf.mean, -1.0/3.0))
        self.assertTrue(np.isclose(leaf.std, np.sqrt(1/3*np.sum((np.array([[0.1], [-1.8], [0.7]])+1.0/3.0)**2))))

    def test_mle_nan_strategy_callable(self):

        leaf = LogNormal(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, np.array([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = LogNormal(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, np.array([[np.nan], [1], [0], [1]]), nan_strategy=1)

    def test_weighted_mle(self):

        leaf = LogNormal(Scope([0]))

        data = np.vstack([
            np.random.lognormal(1.7, 0.8, size=(10000,1)),
            np.random.lognormal(0.5, 1.4, size=(10000,1))
        ])
        weights = np.concatenate([
            np.zeros(10000),
            np.ones(10000)
        ])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.isclose(leaf.mean, 0.5, atol=1e-2, rtol=1e-1))
        self.assertTrue(np.isclose(leaf.std, 1.4, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()