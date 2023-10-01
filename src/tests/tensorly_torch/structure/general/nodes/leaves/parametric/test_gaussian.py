import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Exponential distribution: mean in R, std > 0

    mean = random.random()

    # mean = inf and mean = nan
    tc.assertRaises(Exception, Gaussian, Scope([0]), np.inf, 1.0)
    tc.assertRaises(Exception, Gaussian, Scope([0]), -np.inf, 1.0)
    tc.assertRaises(Exception, Gaussian, Scope([0]), np.nan, 1.0)

    # std = 0 and std < 0
    tc.assertRaises(Exception, Gaussian, Scope([0]), mean, 0.0)
    tc.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nextafter(0.0, -1.0))
    # std = inf and std = nan
    tc.assertRaises(Exception, Gaussian, Scope([0]), mean, np.inf)
    tc.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Gaussian, Scope([]), 0.0, 1.0)
    tc.assertRaises(Exception, Gaussian, Scope([0, 1]), 0.0, 1.0)
    tc.assertRaises(Exception, Gaussian, Scope([0], [1]), 0.0, 1.0)

def test_structural_marginalization(do_for_all_backends):

    gaussian = Gaussian(Scope([0]), 0.0, 1.0)

    tc.assertTrue(marginalize(gaussian, [1]) is not None)
    tc.assertTrue(marginalize(gaussian, [0]) is None)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # Gaussian feature type class
    tc.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]))

    # Gaussian feature type instance
    tc.assertTrue(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)])]))

    # invalid feature type
    tc.assertFalse(Gaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # conditional scope
    tc.assertFalse(Gaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        Gaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Continuous])])
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.mean), tl.tensor(0.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.std), tl.tensor(1.0)))

    gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])])
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.mean), tl.tensor(0.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.std), tl.tensor(1.0)))

    gaussian = Gaussian.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(-1.0, 1.5)])])
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.mean), tl.tensor(-1.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.std), tl.tensor(1.5)))
    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Gaussian.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Gaussian.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Gaussian.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        GaussianInst = GaussianBase
    elif tl.get_backend() == "pytorch":
        GaussianInst = GaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Gaussian))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Gaussian,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gaussian = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Gaussian(mean=-1.0, std=0.5)])])
    tc.assertTrue(isinstance(gaussian, GaussianInst))
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.mean), tl.tensor(-1.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gaussian.std), tl.tensor(0.5)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero
    gaussian = Gaussian(Scope([0]), mean, std)
    for backend in backends:
        with tl.backend_context(backend):
            gaussian_updated = updateBackend(gaussian)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*gaussian.get_params()]),
                    np.array([*gaussian_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
