import random
import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_stack

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondMultivariateGaussian
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext


class TestMultivariateGaussian(unittest.TestCase):
    def test_likelihood_no_mean(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]), cond_f=lambda data: {"cov": tl.eye(2)})
        self.assertRaises(
            KeyError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0], [1]]),
        )

    def test_likelihood_no_cov(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]), cond_f=lambda data: {"mean": tl.zeros(2)})
        self.assertRaises(
            KeyError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0, 0], [1, 1]]),
        )

    def test_likelihood_no_mean_std(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0, 0], [1, 1]]),
        )

    def test_likelihood_module_cond_f(self):

        cond_f = lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)}

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]), cond_f=cond_f)

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {
            "mean": tl.zeros(2),
            "cov": tl.eye(2),
        }

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_args_cond_f(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

        cond_f = lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)}

        dispatch_ctx = DispatchContext()
        dispatch_ctx.args[multivariate_gaussian] = {"cond_f": cond_f}

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
        log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_1(self):

        # ----- configuration 1 -----
        mean = tl.zeros(2)
        cov = tl.eye(2)

        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]), cond_f=lambda data: {"mean": mean, "cov": cov}
        )

        # create test inputs/outputs
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)
        targets = tl.tensor([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        mean = tl.arange(3)
        cov = tl.tensor(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ]
        )

        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1, 2], [3]),
            cond_f=lambda data: {"mean": mean, "cov": cov},
        )

        # create test inputs/outputs
        data = tl_stack(
            [
                mean,
                tl.ones(3),
                -tl.ones(3),
            ],
            axis=0,
        )

        targets = tl.tensor([[0.0366580], [0.0159315], [0.0081795]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, targets))

    def test_likelihood_mean_none(self):

        # dummy distribution and data
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"mean": None, "cov": tl.eye(2)},
        )
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)

        self.assertRaises(Exception, likelihood, multivariate_gaussian, data)

    def test_likelihood_cov_none(self):

        # dummy distribution and data
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"mean": tl.zeros(2), "cov": None},
        )
        data = tl_stack([tl.zeros(2), tl.ones(2)], axis=0)

        # set parameters to None manually
        multivariate_gaussian.cov = None
        self.assertRaises(Exception, likelihood, multivariate_gaussian, data)

    def test_likelihood_marginalization(self):

        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)},
        )
        data = tl.tensor([[tl.nan, tl.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(tl_allclose(probs, tl.exp(log_probs)))
        self.assertTrue(tl_allclose(probs, 1.0))

        # check partial marginalization
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[tl.nan, 0.0]]),
        )

    def test_support(self):

        # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1], [2]),
            cond_f=lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)},
        )

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[-tl.inf, 0.0]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            tl.tensor([[0.0, tl.inf]]),
        )


if __name__ == "__main__":
    unittest.main()
