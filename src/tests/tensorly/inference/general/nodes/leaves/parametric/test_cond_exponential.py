import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondExponential
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondExponential(unittest.TestCase):
    def test_likelihood_no_l(self):

        exponential = CondExponential(Scope([0], [1]))
        self.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"l": 0.5}

        exponential = CondExponential(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_l(self):

        exponential = CondExponential(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"l": 0.5}

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        exponential = CondExponential(Scope([0], [1]))

        cond_f = lambda data: {"l": 0.5}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[exponential] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 0.5})

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[1.0], [0.135335], [0.00673795]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.5})

        # create test inputs/outputs
        data = tl.tensor([[0], [2], [5]])
        targets = tl.tensor([[1.5], [0.0746806], [0.000829627]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_l_none(self):

        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": None})

        data = tl.tensor([[0], [2], [5]])

        # set parameter to None manually
        exponential.l = None
        self.assertRaises(Exception, likelihood, exponential, data)

    def test_likelihood_marginalization(self):

        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
        data = tl.tensor([[tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Exponential distribution: floats [0,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.5})

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[tl.inf]]))

        # check valid float values (within range)
        log_likelihood(exponential, tl.tensor([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(exponential, tl.tensor([[10.5]]))

        # check invalid float values (outside range)
        self.assertRaises(
            ValueError,
            log_likelihood,
            exponential,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )

        # edge case 0
        data = tl.tensor([[0.0]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(all(probs != 0.0))
        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))


if __name__ == "__main__":
    unittest.main()
