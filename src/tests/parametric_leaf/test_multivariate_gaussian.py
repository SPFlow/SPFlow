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

        multivariate_gaussian = MultivariateGaussian([0,1], np.zeros(2), np.eye(2))
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        # set parameters to None manually
        multivariate_gaussian.covariance_matrix = None
        self.assertRaises(Exception, likelihood, SPN(), multivariate_gaussian, data)
        multivariate_gaussian.mean_vector = None
        self.assertRaises(Exception, likelihood, SPN(), multivariate_gaussian, data)

        # invalid scope lengths
        self.assertRaises(Exception, MultivariateGaussian, [], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertRaises(Exception, MultivariateGaussian, [0,1,2], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
