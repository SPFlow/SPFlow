import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_poisson import CondPoisson as CondPoissonBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_poisson import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_poisson import CondPoisson as CondPoissonTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_poisson import CondPoisson
from spflow.tensorly.structure.spn.nodes.sum_node import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    poisson = CondPoisson(Scope([0], [1]))
    tc.assertTrue(poisson.cond_f is None)
    poisson = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
    tc.assertTrue(isinstance(poisson.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondPoisson, Scope([]))
    tc.assertRaises(Exception, CondPoisson, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondPoisson, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)
    torch.set_default_dtype(torch.float64)

    poisson = CondPoisson(Scope([0], [1]))

    # l = 0
    poisson.set_cond_f(lambda data: {"l": tl.tensor(0.0, dtype=tl.float32)})
    tc.assertTrue(poisson.retrieve_params(np.array([[1.0]]), DispatchContext()) == tl.tensor(0.0))
    # l > 0
    poisson.set_cond_f(lambda data: {"l": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32))})
    tc.assertTrue(
        poisson.retrieve_params(np.array([[1.0]]), DispatchContext())
        == tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32))
    )
    # l = -inf and l = inf
    poisson.set_cond_f(lambda data: {"l": tl.tensor(float("inf"))})
    tc.assertRaises(
        Exception,
        poisson.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    poisson.set_cond_f(lambda data: {"l": -tl.tensor(float("inf"))})
    tc.assertRaises(
        Exception,
        poisson.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # l = nan
    poisson.set_cond_f(lambda data: {"l": tl.tensor(float("nan"))})
    tc.assertRaises(
        Exception,
        poisson.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(CondPoisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # Poisson feature type class
    tc.assertTrue(CondPoisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])]))

    # Poisson feature type instance
    tc.assertTrue(CondPoisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(1.0)])]))

    # invalid feature type
    tc.assertFalse(CondPoisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # non-conditional scope
    tc.assertFalse(CondPoisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        CondPoisson.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    poisson = CondPoisson.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])])
    poisson = CondPoisson.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])])
    poisson = CondPoisson.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(l=1.5)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondPoisson.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondPoisson.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondPoisson.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondPoissonInst = CondPoissonBase
    elif tl.get_backend() == "pytorch":
        CondPoissonInst = CondPoissonTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondPoisson))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondPoisson,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    poisson = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(l=1.5)])])
    tc.assertTrue(isinstance(poisson, CondPoissonInst))

def test_structural_marginalization(do_for_all_backends):

    poisson = CondPoisson(Scope([0], [1]))

    tc.assertTrue(marginalize(poisson, [1]) is not None)
    tc.assertTrue(marginalize(poisson, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_poisson = CondPoisson(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_poisson_updated = updateBackend(cond_poisson)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_poisson.scopes_out == cond_poisson_updated.scopes_out))


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    model_default = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(param, float))
    else:
        tc.assertTrue(param.dtype == tl.float32)

    # change to float64 model
    model_updated = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
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
    model_default = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
    model_updated = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
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
