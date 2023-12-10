import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.poisson import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.poisson import Poisson as PoissonBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.poisson import Poisson as PoissonTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_poisson import Poisson
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)

    # l = 0
    Poisson(Scope([0]), 0.0)
    # l > 0
    Poisson(Scope([0]), np.nextafter(tl.tensor(0.0), tl.tensor(1.0)))
    # l = -inf and l = inf
    tc.assertRaises(Exception, Poisson, Scope([0]), -np.inf)
    tc.assertRaises(Exception, Poisson, Scope([0]), np.inf)
    # l = nan
    tc.assertRaises(Exception, Poisson, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Poisson, Scope([]), 1)
    tc.assertRaises(Exception, Poisson, Scope([0, 1]), 1)
    tc.assertRaises(Exception, Poisson, Scope([0], [1]), 1)

def test_structural_marginalization(do_for_all_backends):

    poisson = Poisson(Scope([0]), 1.0)

    tc.assertTrue(marginalize(poisson, [1]) is not None)
    tc.assertTrue(marginalize(poisson, [0]) is None)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Poisson feature type class
    tc.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Poisson])]))

    # Poisson feature type instance
    tc.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Poisson(1.0)])]))

    # invalid feature type
    tc.assertFalse(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(Poisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        Poisson.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
    tc.assertTrue(np.isclose(tl_toNumpy(poisson.l), tl.tensor(1.0)))

    poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Poisson])])
    tc.assertTrue(np.isclose(tl_toNumpy(poisson.l), tl.tensor(1.0)))

    poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])])
    tc.assertTrue(np.isclose(tl_toNumpy(poisson.l), tl.tensor(1.5)))

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Poisson.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Poisson.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Poisson.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        PoissonInst = PoissonBase
    elif tl.get_backend() == "pytorch":
        PoissonInst = PoissonTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Poisson))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Poisson,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Poisson])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    poisson = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])])
    tc.assertTrue(isinstance(poisson, PoissonInst))
    tc.assertTrue(np.isclose(tl_toNumpy(poisson.l), tl.tensor(1.5)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.random()
    poisson = Poisson(Scope([0]), l)
    for backend in backends:
        with tl.backend_context(backend):
            poisson_updated = updateBackend(poisson)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*poisson.get_params()]),
                    np.array([*poisson_updated.get_params()]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    l = random.random()
    model_default = Poisson(Scope([0]), l)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.l, float))
    else:
        tc.assertTrue(model_default.l.dtype == tl.float32)

    # change to float64 model
    model_updated = Poisson(Scope([0]), l)
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
    model_default = Poisson(Scope([0]), l)
    model_updated = Poisson(Scope([0]), l)
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
