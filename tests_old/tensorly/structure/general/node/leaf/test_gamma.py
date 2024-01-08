import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.node.leaf.binomial import updateBackend
from spflow.base.structure.general.node.leaf.gamma import Gamma as GammaBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.autoleaf import AutoLeaf
from spflow.torch.structure.general.node.leaf.gamma import Gamma as GammaTorch
from spflow.modules.node import Gamma
from spflow.torch.structure import marginalize
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_initialization(do_for_all_backends):
    # Valid parameters for Gamma distribution: alpha>0, beta>0

    # alpha > 0
    Gamma(
        Scope([0]),
        tle.nextafter(tl.tensor(0.0), tl.tensor(1.0)),
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
        tle.nextafter(tl.tensor(0.0), tl.tensor(1.0)),
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
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.alpha), tl.tensor(1.0)))
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.beta), tl.tensor(1.0)))

    gamma = Gamma.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gamma])])
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.alpha), tl.tensor(1.0)))
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.beta), tl.tensor(1.0)))

    gamma = Gamma.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.5, 0.5)])])
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.alpha), tl.tensor(1.5)))
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.beta), tl.tensor(0.5)))

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
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.alpha), tl.tensor(1.5)))
    tc.assertTrue(np.isclose(tle.toNumpy(gamma.beta), tl.tensor(0.5)))


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


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    alpha = random.uniform(1, 5)
    beta = random.uniform(1, 5)
    model_default = Gamma(Scope([0]), alpha, beta)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.alpha, float))
        tc.assertTrue(isinstance(model_default.beta, float))
    else:
        tc.assertTrue(model_default.alpha.dtype == tl.float32)
        tc.assertTrue(model_default.beta.dtype == tl.float32)

    # change to float64 model
    model_updated = Gamma(Scope([0]), alpha, beta)
    model_updated.to_dtype(tl.float64)
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_updated.alpha, float))
        tc.assertTrue(isinstance(model_updated.beta, float))
    else:
        tc.assertTrue(model_updated.alpha.dtype == tl.float64)
        tc.assertTrue(model_updated.beta.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    alpha = random.uniform(1, 5)
    beta = random.uniform(1, 5)
    torch.set_default_dtype(torch.float32)
    model_default = Gamma(Scope([0]), alpha, beta)
    model_updated = Gamma(Scope([0]), alpha, beta)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(model_default.alpha.device.type == "cpu")
    tc.assertTrue(model_updated.alpha.device.type == "cuda")

    tc.assertTrue(model_default.beta.device.type == "cpu")
    tc.assertTrue(model_updated.beta.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
