from spflow.base.structure.nodes.leaves.parametric import MultivariateGaussian
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        mean_vector = np.zeros(2)
        covariance_matrix = np.eye(2)

        multivariate_gaussian = MultivariateGaussian(
            [0, 1], mean_vector.tolist(), covariance_matrix.tolist()
        )

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        targets = np.array([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data, SPN())
        log_probs = log_likelihood(multivariate_gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        mean_vector = np.arange(3)
        covariance_matrix = np.array(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ]
        )

        multivariate_gaussian = MultivariateGaussian(
            [0, 1, 2], mean_vector.tolist(), covariance_matrix.tolist()
        )

        # create test inputs/outputs
        data = np.stack(
            [
                mean_vector,
                np.ones(3),
                -np.ones(3),
            ],
            axis=0,
        )

        targets = np.array([[0.0366580], [0.0159315], [0.0081795]])

        probs = likelihood(multivariate_gaussian, data, SPN())
        log_probs = log_likelihood(multivariate_gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite (TODO: PDF only exists if p.d.?)

        # mean contains inf and mean contains nan
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 1], np.array([0.0, np.inf]), np.eye(2)
        )
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 1], np.array([-np.inf, 0.0]), np.eye(2)
        )
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 1], np.array([0.0, np.nan]), np.eye(2)
        )

        # mean vector of wrong shape
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], np.zeros(3), np.eye(2))
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], np.zeros((1, 1, 2)), np.eye(2))

        # covariance matrix of wrong shape
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], np.zeros(2), M)
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], np.zeros(2), M.T)
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], np.zeros(2), np.eye(3))
        # covariance matrix not symmetric positive semi-definite
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 1], np.array([[1.0, 0.0], [1.0, 0.0]])
        )
        self.assertRaises(Exception, MultivariateGaussian, [0, 1], -np.eye(2))
        # covariance matrix containing inf or nan
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 0], np.array([[np.inf, 0], [0, np.inf]])
        )
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 0], np.array([[np.nan, 0], [0, np.nan]])
        )

        # dummy distribution and data
        multivariate_gaussian = MultivariateGaussian([0, 1], np.zeros(2), np.eye(2))
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        # set parameters to None manually
        multivariate_gaussian.covariance_matrix = None
        self.assertRaises(Exception, likelihood, multivariate_gaussian, data, SPN())
        multivariate_gaussian.mean_vector = None
        self.assertRaises(Exception, likelihood, multivariate_gaussian, data, SPN())

        # invalid scope lengths
        self.assertRaises(Exception, MultivariateGaussian, [], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertRaises(
            Exception, MultivariateGaussian, [0, 1, 2], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
        )

    def test_support(self):

        # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k

        # TODO:
        #   likelihood:     None
        #   log-likelihood: None

        multivariate_gaussian = MultivariateGaussian([0, 1], np.zeros(2), np.eye(2))

        # check infinite values
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[-np.inf, 0.0]]), SPN()
        )
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[0.0, np.inf]]), SPN()
        )

    def test_marginalization(self):

        multivariate_gaussian = MultivariateGaussian([0, 1], np.zeros(2), np.eye(2))
        data = np.array([[np.nan, np.nan]])

        # should not raise and error and should return 1 (0 in log-space)
        probs = likelihood(multivariate_gaussian, data, SPN())
        log_probs = log_likelihood(multivariate_gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, 1.0))

        # check partial marginalization
        self.assertRaises(
            ValueError, log_likelihood, multivariate_gaussian, np.array([[np.nan, 0.0]]), SPN()
        )


if __name__ == "__main__":
    unittest.main()
