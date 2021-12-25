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
        self.assertRaises(Exception, likelihood, geometric, data, SPN())

        # invalid scope lengths
        self.assertRaises(Exception, Geometric, [], 0.5)
        self.assertRaises(Exception, Geometric, [0, 1], 0.5)

    def test_support(self):

        # Support for Geometric distribution: integers N\{0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        geometric = Geometric([0], 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, geometric, np.array([[np.inf]]), SPN())
        self.assertRaises(ValueError, log_likelihood, geometric, np.array([[-np.inf]]), SPN())

        # valid integers, but outside valid range
        self.assertRaises(ValueError, log_likelihood, geometric, np.array([[0.0]]), SPN())

        # valid integers within valid range
        data = np.array([[1], [10]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # invalid floats
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[np.nextafter(1.0, 0.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[np.nextafter(1.0, 2.0)]]), SPN()
        )
        self.assertRaises(ValueError, log_likelihood, geometric, np.array([[1.5]]), SPN())

    def test_marginalization(self):

        geometric = Geometric([0], 0.5)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
