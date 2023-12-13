import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_geometric import CondGeometric as CondGeometricBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_geometric import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_geometric import CondGeometric as CondGeometricTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_geometric import CondGeometric
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialiation(do_for_all_backends):

    geometric = CondGeometric(Scope([0], [1]))
    tc.assertTrue(geometric.cond_f is None)
    geometric = CondGeometric(Scope([0], [1]), lambda x: {"p": 0.5})
    tc.assertTrue(isinstance(geometric.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondGeometric, Scope([]))
    tc.assertRaises(Exception, CondGeometric, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondGeometric, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Geometric distribution: p in (0,1]

    geometric = CondGeometric(Scope([0], [1]))

    # p = 0
    geometric.set_cond_f(lambda data: {"p": 0.0})
    tc.assertRaises(
        Exception,
        geometric.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # p = inf and p = nan
    geometric.set_cond_f(lambda data: {"p": np.inf})
    tc.assertRaises(
        Exception,
        geometric.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    geometric.set_cond_f(lambda data: {"p": np.nan})
    tc.assertRaises(
        Exception,
        geometric.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(CondGeometric.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # Geometric feature type class
    tc.assertTrue(CondGeometric.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])]))

    # Geometric feature type instance
    tc.assertTrue(CondGeometric.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric(0.5)])]))

    # invalid feature type
    tc.assertFalse(CondGeometric.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # non-conditional scope
    tc.assertFalse(CondGeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        CondGeometric.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondGeometric.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])])
    CondGeometric.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])])
    CondGeometric.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric(p=0.75)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGeometric.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGeometric.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGeometric.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondGeometricInst = CondGeometricBase
    elif tl.get_backend() == "pytorch":
        CondGeometricInst = CondGeometricTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGeometric))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondGeometric,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    geometric = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])])
    tc.assertTrue(isinstance(geometric, CondGeometricInst))

def test_structural_marginalization(do_for_all_backends):

    geometric = CondGeometric(Scope([0], [1]))

    tc.assertTrue(marginalize(geometric, [1]) is not None)
    tc.assertTrue(marginalize(geometric, [0]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_geometric = CondGeometric(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_geometric_updated = updateBackend(cond_geometric)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_geometric.scopes_out == cond_geometric_updated.scopes_out))

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    model_default = CondGeometric(Scope([0], [1]), lambda x: {"p": 0.5})
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(param, float))
    else:
        tc.assertTrue(param.dtype == tl.float32)

    # change to float64 model
    model_updated = CondGeometric(Scope([0], [1]), lambda x: {"p": 0.5})
    model_updated.to_dtype(tl.float64)
    param_up = model_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(param_up, float))
    else:
        tc.assertTrue(param_up.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([param]),
            np.array([param_up]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    p = random.random()
    torch.set_default_dtype(torch.float32)
    model_default = CondGeometric(Scope([0], [1]), lambda x: {"p": 0.5})
    model_updated = CondGeometric(Scope([0], [1]), lambda x: {"p": 0.5})
    # put model on gpu
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    model_updated.to_device(cuda)
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    param_up = model_updated.retrieve_params(np.array([[1.0]]), DispatchContext())



    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(param.device.type == "cpu")
    tc.assertTrue(param_up.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([param]),
            np.array([param_up.cpu()]),
        )
    )

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
