import random
import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import CondGamma
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestCondGamma(unittest.TestCase):
    def test_likelihood_no_alpha(self):

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"beta": 1.0})
        self.assertRaises(KeyError, log_likelihood, gamma, np.array([[0], [1]]))

    def test_likelihood_no_beta(self):

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0})
        self.assertRaises(KeyError, log_likelihood, gamma, np.array([[0], [1]]))

    def test_likelihood_no_alpha_beta(self):

        gamma = CondGamma(Scope([0], [1]))
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[0], [1]]))

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

        gamma = CondGamma(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        gamma = CondGamma(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"alpha": 1.0, "beta": 1.0}

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        gamma = CondGamma(Scope([0], [1]))

        cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[gamma] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 2.0, "beta": 2.0})

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.327492], [0.541341], [0.029745]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 2.0, "beta": 1.0})

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_alpha_none(self):

        # dummy distribution and data
        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": None, "beta": 1.0})
        data = np.array([[0.1], [1.0], [3.0]])

        self.assertRaises(Exception, likelihood, gamma, data)

    def test_likelihood_beta_none(self):

        # dummy distribution and data
        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": None})
        data = np.array([[0.1], [1.0], [3.0]])

        self.assertRaises(Exception, likelihood, gamma, data)

    def test_likelihood_marginalization(self):

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Gamma distribution: floats (0,inf)

        # TODO:
        #   likelihood:     x=0 -> POS_EPS (?)
        #   log-likelihood: x=0 -> POS_EPS (?)

        gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 2.0, "beta": 1.0})

        # check infinite values
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[-np.inf]]))
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[np.inf]]))

        # check finite values > 0
        log_likelihood(gamma, np.array([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(gamma, np.array([[10.5]]))

        data = np.array([[np.nextafter(0.0, 1.0)]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(all(data != 0.0))
        self.assertTrue(np.allclose(probs, np.exp(log_probs)))

        # check invalid float values (outside range)
        self.assertRaises(ValueError, log_likelihood, gamma, np.array([[0.0]]))
        self.assertRaises(
            ValueError,
            log_likelihood,
            gamma,
            np.array([[np.nextafter(0.0, -1.0)]]),
        )

        # TODO: 0


if __name__ == "__main__":
    unittest.main()
