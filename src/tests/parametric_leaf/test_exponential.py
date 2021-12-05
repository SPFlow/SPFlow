from spflow.base.structure.nodes.leaves.parametric import Exponential
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestExponential(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        l = 0.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 1.0

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.0], [0.135335], [0.00673795]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 1.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.5], [0.0746806], [0.000829627]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        
        exponential = Exponential([0], np.nextafter(0.0, 1.0))
        self.assertRaises(Exception, Exponential, [0], 0.0)
        self.assertRaises(Exception, Exponential, [0], -1.0)
        self.assertRaises(Exception, Exponential, [0], np.inf)
        self.assertRaises(Exception, Exponential, [0], np.nan)

        # set parameters to None manually
        exponential.l = None
        data = np.array([[0], [2], [5]])
        self.assertRaises(Exception, likelihood, SPN(), exponential, data)

        # invalid scope lengths
        self.assertRaises(Exception, Exponential, [], 1.0)
        self.assertRaises(Exception, Exponential, [0,1], 1.0)

    def test_support(self):

        l = 1.5

        # create test inputs/outputs
        data = np.array([[np.nextafter(0.0, -1.0)], [0.0]])

        exponential = Exponential([0], l)
        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        self.assertTrue(all(probs[1] != 0.0))


if __name__ == "__main__":
    unittest.main()