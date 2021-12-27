from spflow.base.structure.nodes.leaves.parametric import Bernoulli
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest

import random


class TestBernoulli(unittest.TestCase):
    def test_likelihood(self):

        p = random.random()

        bernoulli = Bernoulli([0], p)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli([0], 0.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 1
        bernoulli = Bernoulli([0], 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, Bernoulli, [0], np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Bernoulli, [0], np.nextafter(0.0, -1.0))

        # p = inf and p = nan
        self.assertRaises(Exception, Bernoulli, [0], np.inf)
        self.assertRaises(Exception, Bernoulli, [0], np.nan)

        # set parameters to None manually
        bernoulli.p = None
        self.assertRaises(Exception, likelihood, bernoulli, data, SPN())

        # invalid scope lengths
        self.assertRaises(Exception, Bernoulli, [], 0.5)
        self.assertRaises(Exception, Bernoulli, [0, 1], 0.5)

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        p = random.random()
        bernoulli = Bernoulli([0], p)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[-np.inf]]), SPN())
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[np.inf]]), SPN())

        # check valid integers inside valid range
        log_likelihood(bernoulli, np.array([[0.0], [1.0]]), SPN())

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[-1]]), SPN())
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[2]]), SPN())

        # check invalid float values
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, np.array([[np.nextafter(0.0, -1.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, np.array([[np.nextafter(0.0, 1.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, np.array([[np.nextafter(1.0, 2.0)]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, bernoulli, np.array([[np.nextafter(1.0, 0.0)]]), SPN()
        )
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[0.5]]), SPN())

    def test_marginalization(self):

        bernoulli = Bernoulli([0], random.random())
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
