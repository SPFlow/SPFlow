import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as CondMultivariateGaussianBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as CondMultivariateGaussianTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianTorch
from spflow.base.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianBase

from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import (
    marginalize,
)

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussian(Scope([0], [1]))
    tc.assertTrue(multivariate_gaussian.cond_f is None)
    multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1], [2]),
        lambda x: {"mean": torch.zeros(2), "cov": torch.eye(2)},
    )
    tc.assertTrue(isinstance(multivariate_gaussian.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondMultivariateGaussian, Scope([]))
    tc.assertRaises(Exception, CondMultivariateGaussian, Scope([0, 1]))

def test_retrieve_params(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

    multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

    # mean contains inf and mean contains nan
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor([0.0, float("inf")]),
            "cov": tl.eye(2),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor([-float("inf"), 0.0]),
            "cov": tl.eye(2),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor([0.0, float("na")]),
            "cov": tl.eye(2),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )

    # mean vector of wrong shape
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(3), "cov": tl.eye(2)})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(1, 1, 2), "cov": tl.eye(2)})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )

    # covariance matrix of wrong shape
    M = tl.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(2), "cov": M})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(2), "cov": M.T})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(2), "cov": tl.zeros(3)})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )

    # covariance matrix not symmetric positive semi-definite
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.zeros(2),
            "cov": tl.tensor([[1.0, 0.0], [1.0, 0.0]]),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(lambda data: {"mean": tl.zeros(2), "cov": -tl.eye(2)})
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )

    # covariance matrix containing inf or nan
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.zeros(2),
            "cov": tl.tensor([[float("inf"), 0], [0, float("inf")]]),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )
    multivariate_gaussian.set_cond_f(
        lambda data: {
            "mean": tl.zeros(2),
            "cov": tl.tensor([[float("nan"), 0], [0, float("nan")]]),
        }
    )
    tc.assertRaises(
        Exception,
        multivariate_gaussian.retrieve_params,
        tl.tensor([[1.0]]),
        DispatchContext(),
    )

    # initialize using lists
    multivariate_gaussian.set_cond_f(lambda data: {"mean": [0.0, 0.0], "cov": [[1.0, 0.0], [0.0, 1.0]]})
    if (do_for_all_backends=="numpy"):
        mean, cov = multivariate_gaussian.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        tc.assertTrue(tl.all(mean == tl.tensor([0.0, 0.0])))
        tc.assertTrue(tl.all(cov == tl.tensor([[1.0, 0.0], [0.0, 1.0]])))
    else:
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        tc.assertTrue(cov_tril is None)
        tc.assertTrue(tl.all(mean == tl.tensor([0.0, 0.0])))
        tc.assertTrue(tl.all(cov == tl.tensor([[1.0, 0.0], [0.0, 1.0]])))

    # initialize using numpy arrays
    if (do_for_all_backends == "numpy"):
        mean, cov = multivariate_gaussian.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        tc.assertTrue(tl.all(mean == tl.zeros(2)))
        tc.assertTrue(tl.all(cov == tl.eye(2)))
    else:
        multivariate_gaussian.set_cond_f(lambda data: {"mean": np.zeros(2), "cov": np.eye(2)})
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        tc.assertTrue(cov_tril is None)
        tc.assertTrue(tl.all(mean == tl.zeros(2)))
        tc.assertTrue(tl.all(cov == tl.eye(2)))

def test_accept(do_for_all_backends):

    # continuous meta types
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertFalse(
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
    tc.assertFalse(
        CondMultivariateGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

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
    tc.assertRaises(
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
    tc.assertRaises(
        ValueError,
        CondMultivariateGaussian.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondMultivariateGaussianInst = CondMultivariateGaussianBase
    elif tl.get_backend() == "pytorch":
        CondMultivariateGaussianInst = CondMultivariateGaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondMultivariateGaussian))

    # make sure leaf is correctly inferred
    tc.assertEqual(
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
    tc.assertTrue(isinstance(multivariate_gaussian, CondMultivariateGaussianInst))

def test_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondMultivariateGaussianInst = CondMultivariateGaussianBase
        CondGaussianInst = CondGaussianBase
    elif tl.get_backend() == "pytorch":
        CondMultivariateGaussianInst = CondMultivariateGaussianTorch
        CondGaussianInst = CondGaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [3]))

    tc.assertTrue(
        isinstance(
            marginalize(multivariate_gaussian, [2]),
            CondMultivariateGaussianInst,
        )
    )
    tc.assertTrue(isinstance(marginalize(multivariate_gaussian, [1]), CondGaussianInst))
    tc.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1, 2], [3]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_multivariate_gaussian_updated = updateBackend(cond_multivariate_gaussian)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_multivariate_gaussian.scopes_out == cond_multivariate_gaussian_updated.scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
