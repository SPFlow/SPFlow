from spflow.base.structure.nodes.leaves.parametric import Gaussian
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest
import random
import math


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):

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

        # ----- invalid parameters -----
        mean = random.random()

        self.assertRaises(Exception, Gaussian, [0], mean, 0.0)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, Gaussian, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, [0], np.nan, 1.0)
        self.assertRaises(Exception, Gaussian, [0], mean, np.inf)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nan)

        # set parameters to None manually
        gaussian.stdev = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)
        gaussian.mean = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)

        # invalid scope length
        self.assertRaises(Exception, Gaussian, [], 0.0, 1.0)