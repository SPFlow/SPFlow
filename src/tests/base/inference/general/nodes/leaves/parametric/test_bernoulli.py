import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Bernoulli
from spflow.meta.data import Scope


class TestBernoulli(unittest.TestCase):
    def test_likelihood(self):

        p = random.random()

        bernoulli = Bernoulli(Scope([0]), p)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_0(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_none(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        data = np.array([[0.0], [1.0]])

        # set parameter to None manually
        bernoulli.p = None
        self.assertRaises(Exception, likelihood, bernoulli, data)

    def test_likelihood_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), random.random())
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        p = random.random()
        bernoulli = Bernoulli(Scope([0]), p)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[np.inf]]))

        # check valid integers inside valid range
        log_likelihood(bernoulli, np.array([[0.0], [1.0]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[-1]]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[2]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            np.array([[np.nextafter(0.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            np.array([[np.nextafter(0.0, 1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            np.array([[np.nextafter(1.0, 2.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            np.array([[np.nextafter(1.0, 0.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, bernoulli, np.array([[0.5]]))


if __name__ == "__main__":
    unittest.main()
