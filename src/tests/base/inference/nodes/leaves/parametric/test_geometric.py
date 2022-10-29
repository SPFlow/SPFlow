from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestGeometric(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        p = 0.2

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        p = 0.5

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        p = 0.8

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_none(self):

        # dummy distribution and data
        geometric = Geometric(Scope([0]), 0.5)
        data = np.array([[1], [5], [10]])

        # set parameter to None manually
        geometric.p = None
        self.assertRaises(Exception, likelihood, geometric, data)

    def test_likelihood_marginalization(self):

        geometric = Geometric(Scope([0]), 0.5)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Geometric distribution: integers N\{0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        geometric = Geometric(Scope([0]), 0.5)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[-np.inf]])
        )

        # valid integers, but outside valid range
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[0.0]])
        )

        # valid integers within valid range
        data = np.array([[1], [10]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # invalid floats
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            np.array([[np.nextafter(1.0, 0.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            np.array([[np.nextafter(1.0, 2.0)]]),
        )
        self.assertRaises(
            ValueError, log_likelihood, geometric, np.array([[1.5]])
        )


if __name__ == "__main__":
    unittest.main()
