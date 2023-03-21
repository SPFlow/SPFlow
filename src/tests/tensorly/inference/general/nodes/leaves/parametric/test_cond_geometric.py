import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondGeometric
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondGeometric(unittest.TestCase):
    def test_likelihood_no_p(self):

        geometric = CondGeometric(Scope([0], [1]))
        self.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": 0.5}

        geometric = CondGeometric(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_p(self):

        geometric = CondGeometric(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"p": 0.5}

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        geometric = CondGeometric(Scope([0], [1]))

        cond_f = lambda data: {"p": 0.5}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[geometric] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.2})

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.8})

        # create test inputs/outputs
        data = tl.tensor([[1], [5], [10]])
        targets = tl.tensor([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        # dummy distribution and data
        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": None})
        data = tl.tensor([[1], [5], [10]])

        self.assertRaises(Exception, likelihood, geometric, data)

    def test_likelihood_marginalization(self):

        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})
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

        geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})

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
