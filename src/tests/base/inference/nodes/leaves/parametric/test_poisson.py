from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.base.inference.nodes.leaves.parametric.poisson import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random


class TestPoisson(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        l = 1

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        l = 4

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[2], [4], [10]])
        targets = np.array([[0.146525], [0.195367], [0.00529248]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        l = 10

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0378333], [0.12511], [0.0347181]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_l_none(self):

        # dummy distribution and data
        poisson = Poisson(Scope([0]), 1)
        data = np.array([[0], [2], [5]])

        # set parameter to None manually
        poisson.l = None
        self.assertRaises(Exception, likelihood, poisson, data)

    def test_likelihood_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)
        data = np.array([[np.nan, np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Poisson distribution: integers N U {0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        l = random.random()

        poisson = Poisson(Scope([0]), l)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, poisson, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, poisson, np.array([[np.inf]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, poisson, np.array([[-1]]))

        # check valid integers within valid range
        log_likelihood(poisson, np.array([[0]]))
        log_likelihood(poisson, np.array([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError, log_likelihood, poisson, np.array([[np.nextafter(0.0, -1.0)]])
        )
        self.assertRaises(
            ValueError, log_likelihood, poisson, np.array([[np.nextafter(0.0, 1.0)]])
        )
        self.assertRaises(ValueError, log_likelihood, poisson, np.array([[10.1]]))


if __name__ == "__main__":
    unittest.main()
