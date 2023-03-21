import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_unsqueeze

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Binomial
from spflow.meta.data import Scope


class TestBinomial(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.5

        binomial = Binomial(Scope([0]), n, p)

        # create test inputs/outputs
        data = tl.tensor([[0], [5], [10]])
        targets = tl.tensor([[0.000976563], [0.246094], [0.000976563]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        n = 5
        p = 0.8

        binomial = Binomial(Scope([0]), n, p)

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.00032], [0.0512], [0.32768]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        n = 15
        p = 0.3

        binomial = Binomial(Scope([0]), n, p)

        # create test inputs/outputs
        data = tl.tensor([[0], [7], [15]])
        targets = tl.tensor([[0.00474756], [0.08113], [0.0000000143489]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_0(self):

        # p = 0
        binomial = Binomial(Scope([0]), 1, 0.0)

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        binomial = Binomial(Scope([0]), 1, 1.0)

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_n_0(self):

        # n = 0
        binomial = Binomial(Scope([0]), 0, 0.5)

        data = tl.tensor([[0.0]])
        targets = tl.tensor([[1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        binomial = Binomial(Scope([0]), 1, 0.5)

        data = tl.tensor([[0.0]])

        # set parameter to None manually
        binomial.p = None
        self.assertRaises(Exception, likelihood, binomial, data)

    def test_likelihood_n_none(self):

        binomial = Binomial(Scope([0]), 1, 0.5)

        data = tl.tensor([[0.0]])

        # set parameter to None manually
        binomial.n = None
        self.assertRaises(Exception, likelihood, binomial, data)

    def test_likelihood_marginalization(self):

        binomial = Binomial(Scope([0]), 5, 0.5)
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        binomial = Binomial(Scope([0]), 5, 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[tl.inf]]))

        # check valid integers inside valid range
        log_likelihood(binomial, tl_unsqueeze(tl.tensor(list(range(binomial.n + 1))), 1))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[-1]]))
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[binomial.n + 1]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            tl.tensor([[np.nextafter(0.0, 1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            tl.tensor([[np.nextafter(binomial.n, binomial.n + 1)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            tl.tensor([[np.nextafter(binomial.n, 0.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[0.5]]))
        self.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[3.5]]))


if __name__ == "__main__":
    unittest.main()
