import math
import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Gaussian
from spflow.meta.data import Scope


class TestGaussian(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- unit variance -----
        mean = random.random()
        var = 1.0

        gaussian = Gaussian(Scope([0]), mean, math.sqrt(var))

        # create test inputs/outputs
        data = tl.tensor([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = tl.tensor([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- larger variance -----
        mean = random.random()
        var = 5.0

        gaussian = Gaussian(Scope([0]), mean, math.sqrt(var))

        # create test inputs/outputs
        data = tl.tensor([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = tl.tensor([[0.178412], [0.108212], [0.108212]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- smaller variance -----
        mean = random.random()
        var = 0.2

        gaussian = Gaussian(Scope([0]), mean, math.sqrt(var))

        # create test inputs/outputs
        data = tl.tensor([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = tl.tensor([[0.892062], [0.541062], [0.541062]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_mean_none(self):

        # dummy distribution and data
        gaussian = Gaussian(Scope([0]), 0.0, 1.0)
        data = np.random.randn(1, 3)

        # set parameter to None manually
        gaussian.mean = None
        self.assertRaises(Exception, likelihood, gaussian, data)

    def test_likelihood_std_none(self):

        # dummy distribution and data
        gaussian = Gaussian(Scope([0]), 0.0, 1.0)
        data = np.random.randn(1, 3)

        # set parameter to None manually
        gaussian.std = None
        self.assertRaises(Exception, likelihood, gaussian, data)

    def test_likelihood_marginalization(self):

        gaussian = Gaussian(Scope([0]), 0.0, 1.0)
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Gaussian distribution: floats (-inf, inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        gaussian = Gaussian(Scope([0]), 0.0, 1.0)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, gaussian, tl.tensor([[tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, gaussian, tl.tensor([[-tl.inf]]))


if __name__ == "__main__":
    unittest.main()
