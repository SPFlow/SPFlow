from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite (TODO: PDF only exists if p.d.?)

        # mean contains inf and mean contains nan
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.array([0.0, np.inf]),
            np.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.array([-np.inf, 0.0]),
            np.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.array([0.0, np.nan]),
            np.eye(2),
        )

        # mean vector of wrong shape
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(3),
            np.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros((1, 1, 2)),
            np.eye(2),
        )

        # covariance matrix of wrong shape
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertRaises(
            Exception, MultivariateGaussian, Scope([0, 1]), np.zeros(2), M
        )
        self.assertRaises(
            Exception, MultivariateGaussian, Scope([0, 1]), np.zeros(2), M.T
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(2),
            np.eye(3),
        )
        # covariance matrix not symmetric positive semi-definite
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(2),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(2),
            -np.eye(2),
        )
        # covariance matrix containing inf or nan
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(2),
            np.array([[np.inf, 0], [0, np.inf]]),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            np.zeros(2),
            np.array([[np.nan, 0], [0, np.nan]]),
        )

        # duplicate scope variables
        self.assertRaises(
            Exception, Scope, [0, 0]
        )  # makes sure that MultivariateGaussian can also not be given a scope with duplicate query variables

        # invalid scopes
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1, 2]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1], [2]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_structural_marginalization(self):

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]), [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
        )

        self.assertTrue(
            isinstance(
                marginalize(multivariate_gaussian, [2]), MultivariateGaussian
            )
        )
        self.assertTrue(
            isinstance(marginalize(multivariate_gaussian, [1]), Gaussian)
        )
        self.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)


if __name__ == "__main__":
    unittest.main()
