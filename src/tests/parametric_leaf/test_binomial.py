from spflow.base.structure.nodes.leaves.parametric import Binomial
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestBinomial(unittest.TestCase):
    def test_binomial(self):

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

        # ----- (invalid) parameters -----

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

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(0.0, -1.0))

        # n = 0
        binomial = Binomial([0], 0, 0.5)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, Binomial, [0], -1, 0.5)

        # TODO: n float

        # inf, nan
        self.assertRaises(Exception, Binomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, Binomial, [0], np.nan, 0.5)
        self.assertRaises(Exception, Binomial, [0], 1, np.inf)
        self.assertRaises(Exception, Binomial, [0], 1, np.nan)

        # set parameters to None manually
        binomial.p = None
        self.assertRaises(Exception, likelihood, SPN(), binomial, data)
        binomial.n = None
        self.assertRaises(Exception, likelihood, SPN(), binomial, data)

        # invalid scope length
        self.assertRaises(Exception, Binomial, [], 1, 0.5)

        # ----- support -----

        binomial = Binomial([0], 1, 0.0)

        data = np.array([[-1.0], [2.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs == 0))


if __name__ == "__main__":
    unittest.main()
