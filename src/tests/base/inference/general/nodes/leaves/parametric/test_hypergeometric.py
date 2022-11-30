import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Hypergeometric
from spflow.meta.data import Scope


class TestHypergeometric(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0257071], [0.147368], [0.0270206]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.00723683], [0.259334], [0.00059342]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_n_none(self):

        # dummy distribution and data
        hypergeometric = Hypergeometric(Scope([0]), N=500, M=100, n=50)
        data = np.array([[5], [10], [15]])

        # set parameter to None manually
        hypergeometric.n = None
        self.assertRaises(Exception, likelihood, hypergeometric, data)

    def test_likelihood_M_none(self):

        # dummy distribution and data
        hypergeometric = Hypergeometric(Scope([0]), N=500, M=100, n=50)
        data = np.array([[5], [10], [15]])

        # set parameter to None manually
        hypergeometric.M = None
        self.assertRaises(Exception, likelihood, hypergeometric, data)

    def test_likelihood_N_none(self):

        # dummy distribution and data
        hypergeometric = Hypergeometric(Scope([0]), N=500, M=100, n=50)
        data = np.array([[5], [10], [15]])

        # set parameter to None manually
        hypergeometric.N = None
        self.assertRaises(Exception, likelihood, hypergeometric, data)

    def test_likelihood_marginalization(self):

        hypergeometric = Hypergeometric(Scope([0]), 15, 10, 10)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Hypergeometric distribution: integers {max(0,n+M-N),...,min(n,M)}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        # case n+M-N > 0
        N = 15
        M = 10
        n = 10

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, np.array([[-np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, np.array([[np.inf]])
        )

        # check valid integers inside valid range
        data = np.array([[max(0, n + M - N)], [min(n, M)]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(all(probs != 0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[max(0, n + M - N) - 1]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[min(n, M) + 1]]),
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[np.nextafter(max(0, n + M - N), 100)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[np.nextafter(max(n, M), -1.0)]]),
        )
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, np.array([[5.5]])
        )

        # case n+M-N
        N = 25

        hypergeometric = Hypergeometric(Scope([0]), N, M, n)

        # check valid integers within valid range
        data = np.array([[max(0, n + M - N)], [min(n, M)]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(all(probs != 0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # check valid integers, but outside of valid range
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[max(0, n + M - N) - 1]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[min(n, M) + 1]]),
        )

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[np.nextafter(max(0, n + M - N), 100)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            hypergeometric,
            np.array([[np.nextafter(max(n, M), -1.0)]]),
        )
        self.assertRaises(
            ValueError, log_likelihood, hypergeometric, np.array([[5.5]])
        )


if __name__ == "__main__":
    unittest.main()
