from spflow.base.structure.nodes.leaves.parametric import Geometric
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestGeometric(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        p = 0.2

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        p = 0.5

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        p = 0.8

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):
        
        # Valid parameters for Geometric distribution: p in [0,1]

        # p = 0
        self.assertRaises(Exception, Geometric, [0], 0.0)
        
        # p = inf and p = nan
        self.assertRaises(Exception, Geometric, [0], np.inf)
        self.assertRaises(Exception, Geometric, [0], np.nan)

        # dummy distribution and data
        geometric = Geometric([0], 0.5)
        data = np.array([[1], [5], [10]])

        # set parameters to None manually
        geometric.p = None
        self.assertRaises(Exception, likelihood, SPN(), geometric, data)

        # invalid scope lengths
        self.assertRaises(Exception, Geometric, [], 0.5)
        self.assertRaises(Exception, Geometric, [0,1], 0.5)

    def test_support(self):

        # Support for Geometric distribution: N\{0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin
        #
        #   outside support -> 0 (or error?)

        geometric = Geometric([0], 0.5)

        # edge cases (-inf,inf), finite values outside N\{0} and values R between the valid integers
        data = np.array([[-np.inf], [0.0], [np.nextafter(1.0, 0.0)], [1.5], [np.inf]])
        targets = np.zeros((5,1))

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, targets))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # valid integers
        data = np.array([[1], [10]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))


if __name__ == "__main__":
    unittest.main()
