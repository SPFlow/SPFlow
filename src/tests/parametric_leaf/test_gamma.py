from spflow.base.structure.nodes.leaves.parametric import Gamma
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestGamma(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        alpha = 1.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        alpha = 2.0
        beta = 2.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.327492], [0.541341], [0.029745]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        alpha = 2.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs

        targets = np.array([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        Gamma([0], np.nextafter(0.0, 1.0), 1.0)
        Gamma([0], 1.0, np.nextafter(0.0, 1.0))

        # alpha < 0
        self.assertRaises(Exception, Gamma, [0], np.nextafter(0.0, -1.0), 1.0)
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, Gamma, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gamma, [0], np.nan, 1.0)

        # beta < 0
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nextafter(0.0, -1.0))
        # beta = inf and beta = nan
        self.assertRaises(Exception, Gamma, [0], 1.0, np.inf)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nan)

        # dummy distribution and data
        gamma = Gamma([0], 1.0, 1.0)
        data = np.array([[0.1], [1.0], [3.0]])

        # set parameters to None manually
        gamma.beta = None
        self.assertRaises(Exception, likelihood, gamma, data, SPN())
        gamma.alpha = None
        self.assertRaises(Exception, likelihood, gamma, data, SPN())

        # invalid scope lengths
        self.assertRaises(Exception, Gamma, [], 1.0, 1.0)
        self.assertRaises(Exception, Gamma, [0, 1], 1.0, 1.0)

    def test_support(self):

        # Support for Gamma distribution: floats (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)

        gamma = Gamma([0], 1.0, 1.0)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[-np.inf]]), SPN())
        log_likelihood(gamma, np.array([[np.inf]]), SPN())

        # check finite values > 0
        log_likelihood(gamma, np.array([[np.nextafter(0.0, 1.0)]]), SPN())
        log_likelihood(gamma, np.array([[10.5]]), SPN())

        data = np.array([[np.nextafter(0.0, 1.0)]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(all(data != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # check invalid float values (outside range)
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[0.0]]), SPN())
        self.assertRaises(
            ValueError, log_likelihood, gamma, np.array([[np.nextafter(0.0, -1.0)]]), SPN()
        )

        # TODO: 0

    def test_marginalization(self):

        gamma = Gamma([0], 1.0, 1.0)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
