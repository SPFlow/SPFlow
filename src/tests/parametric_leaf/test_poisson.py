from spflow.base.structure.nodes.leaves.parametric import Poisson
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest
import random


class TestPoisson(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        l = 1

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 4

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[2], [4], [10]])
        targets = np.array([[0.146525], [0.195367], [0.00529248]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 10

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0378333], [0.12511], [0.0347181]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        self.assertRaises(Exception, Poisson, [0], -np.inf)
        self.assertRaises(Exception, Poisson, [0], np.inf)
        self.assertRaises(Exception, Poisson, [0], np.nan)

        poisson = Poisson([0], 1)
        data = np.array([[0], [2], [5]])

        # set parameters to None manually
        poisson.l = None
        self.assertRaises(Exception, likelihood, SPN(), poisson, data)

        # invalid scope length
        self.assertRaises(Exception, Poisson, [], 1)
        self.assertRaises(Exception, Poisson, [0,1], 1)

    def test_support(self):

        l = random.random()

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[-1.0], [-0.5], [0.0]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.all(probs[:2] == 0))
        self.assertTrue(np.all(probs[-1] != 0))


if __name__ == "__main__":
    unittest.main()
