import random
import unittest

import numpy as np
import torch
import tensorly as tl
from spflow.torch.structure.general.nodes.leaves.parametric.exponential import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.exponential import Exponential as ExponentialBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.exponential import Exponential as ExponentialTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_exponential import Exponential
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()


def test_initialization(do_for_all_backends):

    # Valid parameters for Exponential distribution: l>0

    # l > 0
    Exponential(Scope([0]), tl_nextafter(tl.tensor(0.0), tl.tensor(1.0)))
    # l = 0 and l < 0
    tc.assertRaises(Exception, Exponential, Scope([0]), 0.0)
    tc.assertRaises(
        Exception,
        Exponential,
        Scope([0]),
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
    )
    # l = inf and l = nan
    tc.assertRaises(Exception, Exponential, Scope([0]), np.inf)
    tc.assertRaises(Exception, Exponential, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Exponential, Scope([]), 0.5)
    tc.assertRaises(Exception, Exponential, Scope([0, 1]), 0.5)
    tc.assertRaises(Exception, Exponential, Scope([0], [1]), 0.5)

def test_structural_marginalization(do_for_all_backends):

    exponential = Exponential(Scope([0]), 1.0)

    tc.assertTrue(marginalize(exponential, [1]) is not None)
    tc.assertTrue(marginalize(exponential, [0]) is None)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(Exponential.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # Exponential feature type class
    tc.assertTrue(Exponential.accepts([FeatureContext(Scope([0]), [FeatureTypes.Exponential])]))

    # Exponential feature type instance
    tc.assertTrue(Exponential.accepts([FeatureContext(Scope([0]), [FeatureTypes.Exponential(1.0)])]))

    # invalid feature type
    tc.assertFalse(Exponential.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # conditional scope
    tc.assertFalse(Exponential.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        Exponential.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    exponential = Exponential.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Continuous])])
    tc.assertTrue(np.isclose(tl_toNumpy(exponential.l), tl.tensor(1.0)))

    exponential = Exponential.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Exponential])])
    tc.assertTrue(np.isclose(tl_toNumpy(exponential.l), tl.tensor(1.0)))

    exponential = Exponential.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])])
    tc.assertTrue(np.isclose(tl_toNumpy(exponential.l), tl.tensor(1.5)))

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Exponential.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Exponential.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Exponential.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        ExponentialInst = ExponentialBase
    elif tl.get_backend() == "pytorch":
        ExponentialInst = ExponentialTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Exponential))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Exponential,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Exponential])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    exponential = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])])
    tc.assertTrue(isinstance(exponential, ExponentialInst))
    tc.assertTrue(np.isclose(tl_toNumpy(exponential.l), tl.tensor(1.5)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.random()
    exponential = Exponential(Scope([0]), l)
    for backend in backends:
        with tl.backend_context(backend):
            exponential_updated = updateBackend(exponential)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*exponential.get_params()]),
                    np.array([*exponential_updated.get_params()]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    l = random.random()
    model_default = Exponential(Scope([0]), l)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.l, float))
    else:
        tc.assertTrue(model_default.l.dtype == tl.float32)

    # change to float64 model
    model_updated = Exponential(Scope([0]), l)
    model_updated.to_dtype(tl.float64)
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_updated.l, float))
    else:
        tc.assertTrue(model_updated.l.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    l = random.random()
    torch.set_default_dtype(torch.float32)
    model_default = Exponential(Scope([0]), l)
    model_updated = Exponential(Scope([0]), l)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(model_default.l.device.type == "cpu")
    tc.assertTrue(model_updated.l.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
