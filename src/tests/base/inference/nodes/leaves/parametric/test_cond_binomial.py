from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import CondBinomial
from spflow.base.inference.nodes.leaves.parametric.cond_binomial import log_likelihood
from spflow.base.inference.module import likelihood
import numpy as np

import random
import unittest


class TestCondBinomial(unittest.TestCase):
    def test_likelihood_no_p(self):

        binomial = CondBinomial(Scope([0]), n=1)
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        p = random.random()
        cond_f = lambda data: {'p': p}

        binomial = CondBinomial(Scope([0]), n=1, cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_p(self):

        binomial = CondBinomial(Scope([0]), n=1)

        p = random.random()
        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {'p': p}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        binomial = CondBinomial(Scope([0]), n=1)

        p = random.random()
        cond_f = lambda data: {'p': p}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[binomial] = {'cond_f': cond_f}

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.5

        binomial = CondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000976563], [0.246094], [0.000976563]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):
    
        # ----- configuration 2 -----
        n = 5
        p = 0.8

        binomial = CondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.00032], [0.0512], [0.32768]])

        probs = likelihood(
            binomial,
            data
        )
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        n = 15
        p = 0.3

        binomial = CondBinomial(Scope([0]), n, cond_f=lambda data: {'p': p})

        # create test inputs/outputs
        data = np.array([[0], [7], [15]])
        targets = np.array([[0.00474756], [0.08113], [0.0000000143489]])

        probs = likelihood(
            binomial,
            data
        )
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_0(self):

        # p = 0
        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': 0.0})

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_1(self):

        # p = 1
        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': 1.0})

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_n_0(self):

        # n = 0
        binomial = CondBinomial(Scope([0]), 0, cond_f=lambda data: {'p': 0.5})

        data = np.array([[0.0]])
        targets = np.array([[1.0]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_p_none(self):

        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': None})

        data = np.array([[0.0]])

        # set parameter to None manually
        self.assertRaises(Exception, likelihood, binomial, data)

    def test_likelihood_n_none(self):

        binomial = CondBinomial(Scope([0]), 1, cond_f=lambda data: {'p': 0.5})

        data = np.array([[0.0]])

        # set parameter to None manually
        binomial.n = None
        self.assertRaises(Exception, likelihood, binomial, data)
    
    def test_likelihood_marginalization(self):

        binomial = CondBinomial(Scope([0]), 5, cond_f=lambda data: {'p': 0.5})
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Binomial distribution: integers {0,...,n}

        # TODO:
        #   likelihood:         0->0.000000001, 1.0->0.999999999
        #   log-likelihood: -inf->fmin

        binomial = CondBinomial(Scope([0]), 5, cond_f=lambda data: {'p': 0.5})

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[np.inf]]))

        # check valid integers inside valid range
        log_likelihood(binomial, np.expand_dims(np.array(list(range(binomial.n + 1))), 1))

        # check valid integers, but outside of valid range
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[-1]]))
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[binomial.n + 1]]))

        # check invalid float values
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(0.0, -1.0)]])
        )
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(0.0, 1.0)]])
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            binomial,
            np.array([[np.nextafter(binomial.n, binomial.n + 1)]]),
        )
        self.assertRaises(
            ValueError, log_likelihood, binomial, np.array([[np.nextafter(binomial.n, 0.0)]])
        )
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[0.5]]))
        self.assertRaises(ValueError, log_likelihood, binomial, np.array([[3.5]]))


if __name__ == "__main__":
    unittest.main()
