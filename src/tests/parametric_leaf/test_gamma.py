from spflow.base.structure.nodes.leaves.parametric import Gamma
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestGamma(unittest.TestCase):
    def test_gamma(self):

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
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----
        Gamma([0], np.nextafter(0.0, 1.0), 1.0)
        Gamma([0], 1.0, np.nextafter(0.0, 1.0))
        self.assertRaises(Exception, Gamma, [0], np.nextafter(0.0, -1.0), 1.0)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, Gamma, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gamma, [0], np.nan, 1.0)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.inf)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nan)

        # set parameters to None manually
        gamma.beta = None
        self.assertRaises(Exception, likelihood, SPN(), gamma, data)
        gamma.alpha = None
        self.assertRaises(Exception, likelihood, SPN(), gamma, data)

        # invalid scope length
        self.assertRaises(Exception, Gamma, [], 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
