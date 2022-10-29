from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.base.inference.nodes.leaves.parametric.log_normal import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random


class TestLogNormal(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        mean = 0.0
        std = 0.25

        log_normal = LogNormal(Scope([0]), mean, std)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        mean = 0.0
        std = 0.5

        log_normal = LogNormal(Scope([0]), mean, std)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.610455], [0.797885], [0.38287]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        mean = 0.0
        std = 1.0

        log_normal = LogNormal(Scope([0]), mean, std)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_mean_none(self):

        # dummy distribution and data
        log_normal = LogNormal(Scope([0]), 0.0, 1.0)
        data = np.array([[0.5], [1.0], [1.5]])

        # set parameter to None manually
        log_normal.mean = None
        self.assertRaises(Exception, likelihood, log_normal, data)

    def test_likelihood_std_none(self):

        # dummy distribution and data
        log_normal = LogNormal(Scope([0]), 0.0, 1.0)
        data = np.array([[0.5], [1.0], [1.5]])

        # set parameter to None manually
        log_normal.std = None
        self.assertRaises(Exception, likelihood, log_normal, data)

    def test_likelihood_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Log-Normal distribution: floats (0,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[-np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[np.inf]])
        )

        # invalid float values
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[0]])
        )

        # valid float values
        log_likelihood(log_normal, np.array([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(log_normal, np.array([[4.3]]))


if __name__ == "__main__":
    unittest.main()
