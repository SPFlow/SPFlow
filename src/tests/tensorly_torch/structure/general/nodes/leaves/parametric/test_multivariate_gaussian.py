import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as MultivariateGaussianBase
from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianBase
from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianTorch
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as MultivariateGaussianTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_multivariate_gaussian import MultivariateGaussian
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

    # mean contains inf and mean contains nan
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.tensor([0.0, float("inf")]),
        tl.eye(2),
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.tensor([-float("inf"), 0.0]),
        tl.eye(2),
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.tensor([0.0, float("nan")]),
        tl.eye(2),
    )

    # mean vector of wrong shape
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(3),
        tl.eye(2),
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros((1, 1, 2)),
        tl.eye(2),
    )

    # covariance matrix of wrong shape
    M = tl.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tc.assertRaises(Exception, MultivariateGaussian, Scope([0, 1]), tl.zeros(2), M)
    tc.assertRaises(Exception, MultivariateGaussian, Scope([0, 1]), tl.zeros(2), M.T)
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(2),
        np.eye(3),
    )
    # covariance matrix not symmetric positive semi-definite
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(2),
        tl.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(2),
        -tl.eye(2),
    )
    # covariance matrix containing inf or nan
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(2),
        tl.tensor([[float("inf"), 0], [0, float("inf")]]),
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1]),
        tl.zeros(2),
        tl.tensor([[float("nan"), 0], [0, float("nan")]]),
    )

    # duplicate scope variables
    tc.assertRaises(
        Exception, Scope, [0, 0]
    )  # makes sure that MultivariateGaussian can also not be given a scope with duplicate query variables

    # invalid scopes
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([]),
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1, 2]),
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    tc.assertRaises(
        Exception,
        MultivariateGaussian,
        Scope([0, 1], [2]),
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    # initialize using lists
    MultivariateGaussian(Scope([0, 1]), [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    # initialize using numpy arrays
    MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

def test_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        MultivariateGaussianInst = MultivariateGaussianBase
        GaussianInst = GaussianBase
    elif tl.get_backend() == "pytorch":
        MultivariateGaussianInst = MultivariateGaussianTorch
        GaussianInst = GaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    tc.assertTrue(isinstance(marginalize(multivariate_gaussian, [2]), MultivariateGaussianInst))
    tc.assertTrue(isinstance(marginalize(multivariate_gaussian, [1]), GaussianInst))
    tc.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)

def test_accept(do_for_all_backends):

    # continuous meta types
    tc.assertTrue(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

    # Gaussian feature type class
    tc.assertTrue(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                )
            ]
        )
    )

    # Gaussian feature type instance
    tc.assertTrue(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Gaussian(0.0, 1.0),
                        FeatureTypes.Gaussian(0.0, 1.0),
                    ],
                )
            ]
        )
    )

    # continuous meta and Gaussian feature types
    tc.assertTrue(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Gaussian],
                )
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Continuous],
                )
            ]
        )
    )

    # conditional scope
    tc.assertFalse(
        MultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    multivariate_gaussian = MultivariateGaussian.from_signatures(
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.mean), tl.zeros(2)))
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.cov), tl.eye(2)))

    multivariate_gaussian = MultivariateGaussian.from_signatures(
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
            )
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.mean), tl.zeros(2)))
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.cov), tl.eye(2)))

    multivariate_gaussian = MultivariateGaussian.from_signatures(
        [
            FeatureContext(
                Scope([0, 1]),
                [
                    FeatureTypes.Gaussian(-1.0, 1.5),
                    FeatureTypes.Gaussian(1.0, 0.5),
                ],
            )
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.mean), tl.tensor([-1.0, 1.0])))
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(multivariate_gaussian.cov),
            tl.tensor([[1.5, 0.0], [0.0, 0.5]]),
        )
    )

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        MultivariateGaussian.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Continuous],
            )
        ],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        MultivariateGaussian.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        MultivariateGaussianInst = MultivariateGaussianBase
    elif tl.get_backend() == "pytorch":
        MultivariateGaussianInst = MultivariateGaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(MultivariateGaussian))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        MultivariateGaussian,
        AutoLeaf.infer(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                )
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    multivariate_gaussian = AutoLeaf(
        [
            FeatureContext(
                Scope([0, 1]),
                [
                    FeatureTypes.Gaussian(mean=-1.0, std=1.5),
                    FeatureTypes.Gaussian(mean=1.0, std=0.5),
                ],
            )
        ]
    )
    tc.assertTrue(isinstance(multivariate_gaussian, MultivariateGaussianInst))
    tc.assertTrue(np.allclose(tl_toNumpy(multivariate_gaussian.mean), tl.tensor([-1.0, 1.0])))
    tc.assertTrue(
        np.allclose(
            tl_toNumpy(multivariate_gaussian.cov),
            tl.tensor([[1.5, 0.0], [0.0, 0.5]]),
        )
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])
    multivariateGaussian = MultivariateGaussian(Scope([0, 1, 2]), mean, cov)
    for backend in backends:
        with tl.backend_context(backend):
            multivariateGaussian_updated = updateBackend(multivariateGaussian)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([multivariateGaussian.get_params()[0]]),
                    np.array([multivariateGaussian_updated.get_params()[0]]),
                )
            )
            tc.assertTrue(
                np.allclose(
                    np.array([multivariateGaussian.get_params()[1]]),
                    np.array([multivariateGaussian_updated.get_params()[1]]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
