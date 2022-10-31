from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from spflow.base.inference.nodes.leaves.parametric.cond_log_normal import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestCondLogNormal(unittest.TestCase):
    def test_likelihood_no_mean(self):

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=lambda data: {"std": 1.0})
        self.assertRaises(
            KeyError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_no_std(self):

        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0}
        )
        self.assertRaises(
            KeyError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_no_mean_std(self):

        log_normal = CondLogNormal(Scope([0], [1]))
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[0], [1]])
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        log_normal = CondLogNormal(Scope([0], [1]), cond_f=cond_f)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args(self):

        log_normal = CondLogNormal(Scope([0], [1]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"mean": 0.0, "std": 1.0}

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        log_normal = CondLogNormal(Scope([0], [1]))

        cond_f = lambda data: {"mean": 0.0, "std": 1.0}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[log_normal] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(log_normal, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 0.25}
        )

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 0.5}
        )

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.610455], [0.797885], [0.38287]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_3(self):

        # ----- configuration 3 -----
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_mean_none(self):

        # dummy distribution and data
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": None, "std": 1.0}
        )
        data = np.array([[0.5], [1.0], [1.5]])

        self.assertRaises(Exception, likelihood, log_normal, data)

    def test_likelihood_std_none(self):

        # dummy distribution and data
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": None}
        )
        data = np.array([[0.5], [1.0], [1.5]])

        self.assertRaises(Exception, likelihood, log_normal, data)

    def test_likelihood_marginalization(self):

        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )
        data = np.array([[np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

    def test_support(self):

        # Support for Log-Normal distribution: floats (0,inf)

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda data: {"mean": 0.0, "std": 1.0}
        )

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[-np.inf]])
        )
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[np.inf]])
        )

        # invalid float values
        self.assertRaises(
            ValueError, log_likelihood, log_normal, np.array([[0]])
        )

        # valid float values
        log_likelihood(log_normal, np.array([[np.nextafter(0.0, 1.0)]]))
        log_likelihood(log_normal, np.array([[4.3]]))


if __name__ == "__main__":
    unittest.main()
