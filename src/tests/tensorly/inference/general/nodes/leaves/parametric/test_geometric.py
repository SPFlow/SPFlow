import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Geometric
from spflow.meta.data import Scope


class TestGeometric(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        p = 0.2

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        p = 0.5

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        p = 0.8

        geometric = Geometric(Scope([0]), p)

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        # dummy distribution and data
        geometric = Geometric(Scope([0]), 0.5)
        data = tl.tensor([[1], [5], [10]])

        # set parameter to None manually
        geometric.p = None
        self.assertRaises(Exception, likelihood, geometric, data)

    def test_likelihood_marginalization(self):

        geometric = Geometric(Scope([0]), 0.5)
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Geometric distribution: integers N\{0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        geometric = Geometric(Scope([0]), 0.5)

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[-tl.inf]]))

        # valid integers, but outside valid range
        self.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[0.0]]))

        # valid integers within valid range
        data = tl.tensor([[1], [10]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))

        # invalid floats
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            tl.tensor([[np.nextafter(1.0, 0.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            geometric,
            tl.tensor([[np.nextafter(1.0, 2.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[1.5]]))


if __name__ == "__main__":
    unittest.main()
