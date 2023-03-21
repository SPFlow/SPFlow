import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Gamma
from spflow.meta.data import Scope


class TestGamma(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        alpha = 1.0
        beta = 1.0

        gamma = Gamma(Scope([0]), alpha, beta)

        # create test inputs/outputs
        data = tl.tensor([[0.1], [1.0], [3.0]])
        targets = tl.tensor([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        alpha = 2.0
        beta = 2.0

        gamma = Gamma(Scope([0]), alpha, beta)

        # create test inputs/outputs
        data = tl.tensor([[0.1], [1.0], [3.0]])
        targets = tl.tensor([[0.327492], [0.541341], [0.029745]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        alpha = 2.0
        beta = 1.0

        gamma = Gamma(Scope([0]), alpha, beta)

        # create test inputs/outputs
        data = tl.tensor([[0.1], [1.0], [3.0]])
        targets = tl.tensor([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_alpha_none(self):

        # dummy distribution and data
        gamma = Gamma(Scope([0]), 1.0, 1.0)
        data = tl.tensor([[0.1], [1.0], [3.0]])

        # set parameter to None manually
        gamma.alpha = None
        self.assertRaises(Exception, likelihood, gamma, data)

    def test_likelihood_beta_none(self):

        # dummy distribution and data
        gamma = Gamma(Scope([0]), 1.0, 1.0)
        data = tl.tensor([[0.1], [1.0], [3.0]])

        # set parameter to None manually
        gamma.beta = None
        self.assertRaises(Exception, likelihood, gamma, data)

    def test_likelihood_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Gamma distribution: floats (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[tl.inf]]))

        # check finite values > 0
        log_likelihood(gamma, tl.tensor([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(gamma, tl.tensor([[10.5]]))

        data = tl.tensor([[np.nextafter(0.0, 1.0)]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(all(data != 0.0))
        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))

        # check invalid float values (outside range)
        self.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[0.0]]))
        self.assertRaises(
            ValueError,
            log_likelihood,
            gamma,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )

        # TODO: 0


if __name__ == "__main__":
    unittest.main()
