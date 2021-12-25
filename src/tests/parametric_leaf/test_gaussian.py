from spflow.base.structure.nodes.leaves.parametric import Gaussian
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest
import random
import math


class TestGaussian(unittest.TestCase):
    def test_likelihood(self):

        # ----- unit variance -----
        mean = random.random()
        var = 1.0

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- larger variance -----
        mean = random.random()
        var = 5.0

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.178412], [0.108212], [0.108212]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- smaller variance -----
        mean = random.random()
        var = 0.2

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.892062], [0.541062], [0.541062]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Exponential distribution: mean in R, stdev > 0

        mean = random.random()

        # mean = inf and mean = nan
        self.assertRaises(Exception, Gaussian, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, [0], np.nan, 1.0)

        # stdev = 0 and stdev > 0
        self.assertRaises(Exception, Gaussian, [0], mean, 0.0)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nextafter(0.0, -1.0))
        # stdev = inf and stdev = nan
        self.assertRaises(Exception, Gaussian, [0], mean, np.inf)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nan)

        # dummy distribution and data
        gaussian = Gaussian([0], 0.0, 1.0)
        data = np.random.randn(1, 3)

        # set parameters to None manually
        gaussian.stdev = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)
        gaussian.mean = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)

        # invalid scope lengths
        self.assertRaises(Exception, Gaussian, [], 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, [0, 1], 0.0, 1.0)

    def test_support(self):

        # Support for Gaussian distribution: floats R (TODO: R or (-inf, inf)?)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        gaussian = Gaussian([0], 0.0, 1.0)

        # check infinite values (TODO)
        data = np.array([[-np.inf], [np.inf]])
        targets = np.zeros((2, 1))

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, targets))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

    def test_marginalization(self):

        gaussian = Gaussian([0], 0.0, 1.0)
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))


if __name__ == "__main__":
    unittest.main()
