from spflow.base.structure.nodes.leaves.parametric import Binomial
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestBinomial(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.5

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000976563], [0.246094], [0.000976563]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 5
        p = 0.8

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.00032], [0.0512], [0.32768]])

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        n = 15
        p = 0.3

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [7], [15]])
        targets = np.array([[0.00474756], [0.08113], [0.0000000143489]])

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        # p = 0
        binomial = Binomial([0], 1, 0.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 1
        binomial = Binomial([0], 1, 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(0.0, -1.0))

        # p = inf and p = nan
        self.assertRaises(Exception, Binomial, [0], 1, np.inf)
        self.assertRaises(Exception, Binomial, [0], 1, np.nan)

        # n = 0
        binomial = Binomial([0], 0, 0.5)

        data = np.array([[0.0]])
        targets = np.array([[1.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, Binomial, [0], -1, 0.5)

        # n float
        self.assertRaises(Exception, Binomial, [0], 0.5, 0.5)

        # n = inf and n = nan
        self.assertRaises(Exception, Binomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, Binomial, [0], np.nan, 0.5)

        # set parameters to None manually
        binomial.p = None
        self.assertRaises(Exception, likelihood, binomial, SPN(), data)
        binomial.n = None
        self.assertRaises(Exception, likelihood, binomial, SPN(), data)

        # invalid scope lengths
        self.assertRaises(Exception, Binomial, [], 1, 0.5)
        self.assertRaises(Exception, Binomial, [0, 1], 1, 0.5)

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        binomial = Binomial([0], 5, 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[-np.inf]]), SPN())
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[np.inf]]), SPN())

        # check valid integers inside valid range
        log_likelihood(binomial, np.expand_dims(np.array(list(range(binomial.n + 1))), 1), SPN())

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[-1]]), SPN())
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[binomial.n + 1]]), SPN())

        # check invalid float values
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(0.0, -1.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(0.0, 1.0)]]), SPN()
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            np.array([[np.nextafter(binomial.n, binomial.n + 1)]]),
            SPN(),
        )
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(binomial.n, 0.0)]]), SPN()
        )
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[0.5]]), SPN())
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[3.5]]), SPN())

    def test_marginalization(self):

        binomial = Binomial([0], 5, 0.5)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
