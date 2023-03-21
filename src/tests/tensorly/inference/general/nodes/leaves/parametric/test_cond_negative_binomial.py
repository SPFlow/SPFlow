import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondNegativeBinomial
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondNegativeBinomial(unittest.TestCase):
    def test_likelihood_no_p(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1)
        self.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"p": 1.0}

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=cond_f)

        # create test inputs/outputs
        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_p(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1)

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"p": 1.0}

        # create test inputs/outputs
        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1)

        cond_f = lambda data: {"p": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[negative_binomial] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=10, cond_f=lambda data: {"p": 0.4})

        # create test inputs/outputs
        data = tl.tensor([[0], [5], [10]])
        targets = tl.tensor([[0.000104858], [0.0163238], [0.0585708]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=20, cond_f=lambda data: {"p": 0.3})

        # create test inputs/outputs
        data = tl.tensor([[0], [10], [20]])
        targets = tl.tensor([[0.0000000000348678], [0.0000197282], [0.00191757]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda data: {"p": 1.0})

        data = tl.tensor([[0.0], [1.0]])
        targets = tl.tensor([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_p_none(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda data: {"p": None})

        data = tl.tensor([[0.0], [1.0]])

        self.assertRaises(Exception, likelihood, negative_binomial, data)

    def test_likelihood_n_none(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda data: {"p": 0.5})

        data = tl.tensor([[0.0], [1.0]])

        # set parameter to None manually
        negative_binomial.n = None
        self.assertRaises(Exception, likelihood, negative_binomial, data)

    def test_likelihood_float(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda data: {"p": 0.5})

        # TODO: n float
        self.assertRaises(Exception, likelihood, negative_binomial, 0.5)

    def test_likelihood_marginalization(self):

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=20, cond_f=lambda data: {"p": 0.3})
        data = tl.tensor([[tl.nan, tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

    def test_support(self):

        # Support for Negative Binomial distribution: integers N U {0}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=20, cond_f=lambda data: {"p": 0.3})

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[-tl.inf]]))
        self.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[tl.inf]]))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[-1]]))

        # check valid integers within valid range
        log_likelihood(negative_binomial, tl.tensor([[0]]))
        log_likelihood(negative_binomial, tl.tensor([[100]]))

        # check invalid float values
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            tl.tensor([[np.nextafter(0.0, -1.0)]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            negative_binomial,
            tl.tensor([[np.nextafter(0.0, 1.0)]]),
        )
        self.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[10.1]]))


if __name__ == "__main__":
    unittest.main()
