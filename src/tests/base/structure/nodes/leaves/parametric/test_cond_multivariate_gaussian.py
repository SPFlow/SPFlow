from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
)
from typing import Callable

import numpy as np
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))
        self.assertTrue(multivariate_gaussian.cond_f is None)
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0], [1]),
            cond_f=lambda x: {"mean": np.array([0.0, 0.0]), "cov": np.eye(2)},
        )
        self.assertTrue(isinstance(multivariate_gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([]))
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([0, 1]))

    def test_retrieve_params(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

        # mean contains inf and mean contains nan
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.array([0.0, np.inf]), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.array([-np.inf, 0.0]), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.array([0.0, np.nan]), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )

        # mean vector of wrong shape
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros(3), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros((1, 1, 2)), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.array([0.0, np.nan]), "cov": np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        # covariance matrix of wrong shape
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros(2), "cov": M}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros(2), "cov": M.T}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros(2), "cov": np.eye(3)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        # covariance matrix not symmetric positive semi-definite
        multivariate_gaussian.set_cond_f(
            lambda data: {"mean": np.zeros(2), "cov": -np.eye(2)}
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": np.zeros(2),
                "cov": np.array([[1.0, 0.0], [1.0, 0.0]]),
            }
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        # covariance matrix containing inf or nan
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": np.zeros(2),
                "cov": np.array([[np.inf, 0], [0, np.inf]]),
            }
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )
        multivariate_gaussian.set_cond_f(
            lambda data: {
                "mean": np.zeros(2),
                "cov": np.array([[np.nan, 0], [0, np.nan]]),
            }
        )
        self.assertRaises(
            ValueError,
            multivariate_gaussian.retrieve_params,
            np.array([[1.0, 1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta types
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [
                            FeatureTypes.Gaussian(0.0, 1.0),
                            FeatureTypes.Gaussian(0.0, 1.0),
                        ],
                    )
                ]
            )
        )

        # continuous meta and Gaussian feature types
        self.assertTrue(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondMultivariateGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                )
            ]
        )
        CondMultivariateGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.Gaussian(-1.0, 1.5),
                        FeatureTypes.Gaussian(1.0, 0.5),
                    ],
                )
            ]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondMultivariateGaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Continuous],
                )
            ],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondMultivariateGaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondMultivariateGaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondMultivariateGaussian,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.Gaussian(mean=-1.0, std=1.5),
                        FeatureTypes.Gaussian(mean=1.0, std=0.5),
                    ],
                )
            ]
        )
        self.assertTrue(
            isinstance(multivariate_gaussian, CondMultivariateGaussian)
        )

    def test_structural_marginalization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [3]))

        self.assertTrue(
            isinstance(
                marginalize(multivariate_gaussian, [2]),
                CondMultivariateGaussian,
            )
        )
        self.assertTrue(
            isinstance(marginalize(multivariate_gaussian, [1]), CondGaussian)
        )
        self.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)


if __name__ == "__main__":
    unittest.main()
