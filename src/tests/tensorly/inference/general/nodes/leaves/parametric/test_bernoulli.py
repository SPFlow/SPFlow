import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Bernoulli
from spflow.meta.data import Scope


class TestBernoulli(unittest.TestCase):
    def test_likelihood(self):

        p = random.random()

        bernoulli = Bernoulli(Scope([0]), p)

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_0(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        data = tl.tensor([[0.0], [1.0]])

        # set parameter to None manually
        bernoulli.p = None
        self.assertRaises(Exception, likelihood, bernoulli, data)

    def test_likelihood_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), random.random())
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        p = random.random()
        bernoulli = Bernoulli(Scope([0]), p)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[tl.inf]]))

        # check valid integers inside valid range
        log_likelihood(bernoulli, tl.tensor([[0.0], [1.0]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[-1]]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[2]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            tl.tensor([[np.nextafter(0.0, 1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            tl.tensor([[np.nextafter(1.0, 2.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            bernoulli,
            tl.tensor([[np.nextafter(1.0, 0.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
