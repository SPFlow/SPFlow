from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma
from typing import Callable

import numpy as np
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        gamma = CondGamma(Scope([0]))
        self.assertTrue(gamma.cond_f is None)
        gamma = CondGamma(Scope([0]), cond_f=lambda x: {'alpha': 1.0, 'beta': 1.0})
        self.assertTrue(isinstance(gamma.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGamma, Scope([]))
        self.assertRaises(Exception, CondGamma, Scope([0, 1]))
        self.assertRaises(Exception, CondGamma, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        gamma = CondGamma(Scope([0]))

        # alpha > 0
        gamma.set_cond_f(lambda data: {'alpha': np.nextafter(0.0, 1.0), 'beta': 1.0})
        alpha, beta = gamma.retrieve_params(np.array([[1.0]]), DispatchContext())
        self.assertTrue(alpha == np.nextafter(0.0, 1.0))
        self.assertTrue(beta == 1.0)
        # alpha = 0
        gamma.set_cond_f(lambda data: {'alpha': 0.0, 'beta': 1.0})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        # alpha < 0
        gamma.set_cond_f(lambda data: {'alpha': np.nextafter(0.0, -1.0), 'beta': 1.0})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        # alpha = inf and alpha = nan
        gamma.set_cond_f(lambda data: {'alpha': np.inf, 'beta': 1.0})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        gamma.set_cond_f(lambda data: {'alpha': np.nan, 'beta': 1.0})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
    
        # beta > 0
        gamma.set_cond_f(lambda data: {'alpha': 1.0, 'beta': np.nextafter(0.0, 1.0)})
        alpha, beta = gamma.retrieve_params(np.array([[1.0]]), DispatchContext())
        self.assertTrue(alpha == 1.0)
        self.assertTrue(beta == np.nextafter(0.0, 1.0))
        # beta = 0
        gamma.set_cond_f(lambda data: {'alpha': 1.0, 'beta': 0.0})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        # beta < 0
        gamma.set_cond_f(lambda data: {'alpha': 1.0, 'beta': np.nextafter(0.0, -1.0)})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        # beta = inf and beta = nan
        gamma.set_cond_f(lambda data: {'alpha': 1.0, 'beta': np.inf})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())
        gamma.set_cond_f(lambda data: {'alpha': 1.0, 'beta': np.nan})
        self.assertRaises(ValueError, gamma.retrieve_params, np.array([[1.0]]), DispatchContext())

        # invalid scopes
        self.assertRaises(Exception, CondGamma, Scope([]))
        self.assertRaises(Exception, CondGamma, Scope([0, 1]))
        self.assertRaises(Exception, CondGamma, Scope([0],[1]))

    def test_structural_marginalization(self):

        gamma = CondGamma(Scope([0]))

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)


if __name__ == "__main__":
    unittest.main()
