import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondBernoulli
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondBernoulli(unittest.TestCase):
    def test_likelihood_no_p(self):

        bernoulli = CondBernoulli(Scope([0], [1]))
        self.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        p = random.random()
        cond_f = lambda data: {"p": p}

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_p(self):

        bernoulli = CondBernoulli(Scope([0], [1]))

        p = random.random()
        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"p": p}

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        bernoulli = CondBernoulli(Scope([0], [1]))

        p = random.random()
        cond_f = lambda data: {"p": p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[0], [1]])
        targets = tl.tensor([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_0(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.0})

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 1.0})

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[0.0], [1.0]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": None})

        data = tl.tensor([[0.0], [1.0]])

        # set parameter to None manually
        self.assertRaises(Exception, likelihood, bernoulli, data)

    def test_likelihood_marginalization(self):

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Bernoulli distribution: integers {0,1}

        bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})

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
