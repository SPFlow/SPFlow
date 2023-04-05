import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Poisson
from spflow.meta.data import Scope


class TestPoisson(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        l = 1

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        l = 4

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = tl.tensor([[2], [4], [10]])
        targets = tl.tensor([[0.146525], [0.195367], [0.00529248]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        l = 10

        poisson = Poisson(Scope([0]), l)

        # create test inputs/outputs
        data = tl.tensor([[5], [10], [15]])
        targets = tl.tensor([[0.0378333], [0.12511], [0.0347181]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_l_none(self):

        # dummy distribution and data
        poisson = Poisson(Scope([0]), 1)
        data = tl.tensor([[0], [2], [5]])

        # set parameter to None manually
        poisson.l = None
        self.assertRaises(Exception, likelihood, poisson, data)

    def test_likelihood_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)
        data = tl.tensor([[tl.nan, tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Poisson distribution: integers N U {0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        l = random.random()

        poisson = Poisson(Scope([0]), l)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[tl.inf]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-1]]))

        # check valid integers within valid range
        log_likelihood(poisson, tl.tensor([[0]]))
        log_likelihood(poisson, tl.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            poisson,
            tl.tensor([[np.nextafter(0.0, 1.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[10.1]]))


if __name__ == "__main__":
    unittest.main()
