from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian
from typing import Callable

import numpy as np
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0]))
        self.assertTrue(multivariate_gaussian.cond_f is None)
        multivariate_gaussian = CondMultivariateGaussian(Scope([0]), cond_f=lambda x: {'mean': np.array([0.0, 0.0]), 'cov': np.eye(2)})
        self.assertTrue(isinstance(multivariate_gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([]))
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([0], [1]))
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([0,1],[2]))

    def test_retrieve_params(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

        multivariate_gaussian = CondMultivariateGaussian(Scope([0,1]))

        # mean contains inf and mean contains nan
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.array([0.0, np.inf]), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.array([-np.inf, 0.0]), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.array([0.0, np.nan]), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())

        # mean vector of wrong shape
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(3), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros((1, 1, 2)), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.array([0.0, np.nan]), 'cov': np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        # covariance matrix of wrong shape
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': M})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': M.T})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': np.eye(3)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        # covariance matrix not symmetric positive semi-definite
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': -np.eye(2)})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': np.array([[1.0, 0.0], [1.0, 0.0]])})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        # covariance matrix containing inf or nan
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': np.array([[np.inf, 0], [0, np.inf]])})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': np.array([[np.nan, 0], [0, np.nan]])})
        self.assertRaises(ValueError, multivariate_gaussian.retrieve_params, np.array([[1.0, 1.0]]), DispatchContext())

    def test_structural_marginalization(self):
    
        multivariate_gaussian = CondMultivariateGaussian(Scope([0,1]))

        self.assertTrue(isinstance(marginalize(multivariate_gaussian, [2]), CondMultivariateGaussian))
        self.assertTrue(isinstance(marginalize(multivariate_gaussian, [1]), CondGaussian))
        self.assertTrue(marginalize(multivariate_gaussian, [0,1]) is None)


if __name__ == "__main__":
    unittest.main()
