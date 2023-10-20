import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import Exponential
from spflow.meta.data import Scope


class TestExponential(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        l = 0.5

        exponential = Exponential(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        l = 1.0

        exponential = Exponential(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.0], [0.135335], [0.00673795]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        l = 1.5

        exponential = Exponential(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.5], [0.0746806], [0.000829627]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_l_none(self):

        exponential = Exponential(Scope([0]), 1.0)

        data = np.array([[0], [2], [5]])

        # set parameter to None manually
        exponential.l = None
        self.assertRaises(Exception, likelihood, exponential, data)

    def test_likelihood_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Exponential distribution: floats [0,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        l = 1.5
        exponential = Exponential(Scope([0]), l)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, exponential, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, exponential, np.array([[np.inf]]))

        # check valid float values (within range)
        log_likelihood(exponential, np.array([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(exponential, np.array([[10.5]]))

        # check invalid float values (outside range)
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            np.array([[np.nextafter(0.0, -1.0)]]),
        )

        # edge case 0
        data = np.array([[0.0]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))


if __name__ == "__main__":
    unittest.main()