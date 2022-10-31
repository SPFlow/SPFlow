from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)

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

    def test_accept(self):

        # continuous meta types
        self.assertTrue(MultivariateGaussian.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))]))

        # Gaussian feature type class
        self.assertTrue(MultivariateGaussian.accepts([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1]))]))

        # Gaussian feature type instance
        self.assertTrue(MultivariateGaussian.accepts([([FeatureTypes.Gaussian(0.0, 1.0), FeatureTypes.Gaussian(0.0, 1.0)], Scope([0, 1]))]))

        # continuous meta and Gaussian feature types
        self.assertTrue(MultivariateGaussian.accepts([([FeatureTypes.Continuous, FeatureTypes.Gaussian], Scope([0, 1]))]))

        # invalid feature type
        self.assertFalse(MultivariateGaussian.accepts([([FeatureTypes.Discrete, FeatureTypes.Continuous], Scope([0, 1]))]))

        # conditional scope
        self.assertFalse(MultivariateGaussian.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))]))

        # scope length does not match number of types
        self.assertFalse(MultivariateGaussian.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1, 2]))]))

    def test_initialization_from_signatures(self):

        multivariate_gaussian = MultivariateGaussian.from_signatures([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])
        self.assertTrue(np.all(multivariate_gaussian.mean == np.zeros(2)))
        self.assertTrue(np.all(multivariate_gaussian.cov == np.eye(2)))

        multivariate_gaussian = MultivariateGaussian.from_signatures([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1]))])
        self.assertTrue(np.all(multivariate_gaussian.mean == np.zeros(2)))
        self.assertTrue(np.all(multivariate_gaussian.cov == np.eye(2)))
    
        multivariate_gaussian = MultivariateGaussian.from_signatures([([FeatureTypes.Gaussian(-1.0, 1.5), FeatureTypes.Gaussian(1.0, 0.5)], Scope([0, 1]))])
        self.assertTrue(np.all(multivariate_gaussian.mean == np.array([-1.0, 1.0])))
        self.assertTrue(np.all(multivariate_gaussian.cov == np.array([[1.5, 0.0], [0.0, 0.5]])))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, MultivariateGaussian.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Continuous], Scope([0, 1]))])

        # conditional scope
        self.assertRaises(ValueError, MultivariateGaussian.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, MultivariateGaussian.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(MultivariateGaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(MultivariateGaussian, AutoLeaf.infer([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf([([FeatureTypes.Gaussian(mean=-1.0, std=1.5), FeatureTypes.Gaussian(mean=1.0, std=0.5)], Scope([0, 1]))])
        self.assertTrue(isinstance(multivariate_gaussian, MultivariateGaussian))
        self.assertTrue(np.all(multivariate_gaussian.mean == np.array([-1.0, 1.0])))
        self.assertTrue(np.all(multivariate_gaussian.cov == np.array([[1.5, 0.0], [0.0, 0.5]])))

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
