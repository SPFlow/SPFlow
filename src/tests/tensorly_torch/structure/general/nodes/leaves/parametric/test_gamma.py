import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.binomial import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.gamma import Gamma as GammaBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.gamma import Gamma as GammaTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gamma import Gamma
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Gamma distribution: alpha>0, beta>0

    # alpha > 0
    Gamma(
        Scope([0]),
        tl_nextafter(tl.tensor(0.0), tl.tensor(1.0)),
        1.0,
    )
    # alpha = 0
    tc.assertRaises(Exception, Gamma, Scope([0]), 0.0, 1.0)
    # alpha < 0
    tc.assertRaises(Exception, Gamma, Scope([0]), np.nextafter(0.0, -1.0), 1.0)
    # alpha = inf and alpha = nan
    tc.assertRaises(Exception, Gamma, Scope([0]), np.inf, 1.0)
    tc.assertRaises(Exception, Gamma, Scope([0]), np.nan, 1.0)

    # beta > 0
    Gamma(
        Scope([0]),
        1.0,
        tl_nextafter(tl.tensor(0.0), tl.tensor(1.0)),
    )
    # beta = 0
    tc.assertRaises(Exception, Gamma, Scope([0]), 1.0, 0.0)
    # beta < 0
    tc.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nextafter(0.0, -1.0))
    # beta = inf and beta = non
    tc.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.inf)
    tc.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Gamma, Scope([]), 1.0, 1.0)
    tc.assertRaises(Exception, Gamma, Scope([0, 1]), 1.0, 1.0)
    tc.assertRaises(Exception, Gamma, Scope([0], [1]), 1.0, 1.0)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # Gamma feature type class
    tc.assertTrue(Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gamma])]))

    # Gamma feature type instance
    tc.assertTrue(Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.0, 1.0)])]))

    # invalid feature type
    tc.assertFalse(Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # conditional scope
    tc.assertFalse(Gamma.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        Gamma.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    gamma = Gamma.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Continuous])])
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.alpha), tl.tensor(1.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.beta), tl.tensor(1.0)))

    gamma = Gamma.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gamma])])
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.alpha), tl.tensor(1.0)))
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.beta), tl.tensor(1.0)))

    gamma = Gamma.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.5, 0.5)])])
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.alpha), tl.tensor(1.5)))
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.beta), tl.tensor(0.5)))

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Gamma.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Gamma.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Gamma.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        GammaInst = GammaBase
    elif tl.get_backend() == "pytorch":
        GammaInst = GammaTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Gamma))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Gamma,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Gamma])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gamma = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)])])
    tc.assertTrue(isinstance(gamma, GammaInst))
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.alpha), tl.tensor(1.5)))
    tc.assertTrue(np.isclose(tl_toNumpy(gamma.beta), tl.tensor(0.5)))

def test_structural_marginalization(do_for_all_backends):

    gamma = Gamma(Scope([0]), 1.0, 1.0)

    tc.assertTrue(marginalize(gamma, [1]) is not None)
    tc.assertTrue(marginalize(gamma, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)
    gamma = Gamma(Scope([0]), alpha, beta)
    for backend in backends:
        with tl.backend_context(backend):
            gamma_updated = updateBackend(gamma)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*gamma.get_params()]),
                    np.array([*gamma_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
