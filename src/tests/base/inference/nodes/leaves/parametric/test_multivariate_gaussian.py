from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_likelihood_1(self):

        # ----- configuration 1 -----
        mean = np.zeros(2)
        cov = np.eye(2)

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]), mean.tolist(), cov.tolist()
        )

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        targets = np.array([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_likelihood_2(self):

        # ----- configuration 2 -----
        mean = np.arange(3)
        cov = np.array(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ]
        )

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2]), mean.tolist(), cov.tolist()
        )

        # create test inputs/outputs
        data = np.stack(
            [
                mean,
                np.ones(3),
                -np.ones(3),
            ],
            axis=0,
        )

        targets = np.array([[0.0366580], [0.0159315], [0.0081795]])

        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))
    
    def test_likelihood_mean_none(self):

        # dummy distribution and data
        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        # set parameters to None manually
        multivariate_gaussian.mean = None
        self.assertRaises(Exception, likelihood, multivariate_gaussian, data)

    def test_likelihood_cov_none(self):

        # dummy distribution and data
        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        # set parameters to None manually
        multivariate_gaussian.cov = None
        self.assertRaises(Exception, likelihood, multivariate_gaussian, data)
    
    def test_likelihood_marginalization(self):

        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))
        data = np.array([[np.nan, np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(multivariate_gaussian, data)
        log_probs = log_likelihood(multivariate_gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

        # check partial marginalization
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[np.nan, 0.0]])
        )

    def test_support(self):

        # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k
        multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[-np.inf, 0.0]])
        )
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[0.0, np.inf]])
        )


if __name__ == "__main__":
    unittest.main()
