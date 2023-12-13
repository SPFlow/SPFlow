import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as CondBernoulliBase
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_bernoulli import CondBernoulli
from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import updateBackend
from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as CondBernoulliTorch
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    bernoulli = CondBernoulli(Scope([0], [1]))
    tc.assertTrue(bernoulli.cond_f is None)
    bernoulli = CondBernoulli(Scope([0], [1]), lambda x: {"p": 0.5})
    tc.assertTrue(isinstance(bernoulli.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondBernoulli, Scope([]))
    tc.assertRaises(Exception, CondBernoulli, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondBernoulli, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Bernoulli distribution: p in [0,1]

    bernoulli = CondBernoulli(Scope([0], [1]))

    # p = 0
    bernoulli.set_cond_f(lambda data: {"p": 0.0})
    tc.assertTrue(bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext()) == 0.0)
    # p = 1
    bernoulli.set_cond_f(lambda data: {"p": 1.0})
    tc.assertTrue(bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext()) == 1.0)
    # p < 0 and p > 1
    bernoulli.set_cond_f(lambda data: {"p": tl_nextafter(tl.tensor(1.0), tl.tensor(2.0))})
    tc.assertRaises(
        Exception,
        bernoulli.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    bernoulli.set_cond_f(lambda data: {"p": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0))})
    tc.assertRaises(
        Exception,
        bernoulli.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # p = inf and p = nan
    bernoulli.set_cond_f(lambda data: {"p": np.inf})
    tc.assertRaises(
        Exception,
        bernoulli.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    bernoulli.set_cond_f(lambda data: {"p": np.nan})
    tc.assertRaises(
        Exception,
        bernoulli.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(CondBernoulli.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type class
    tc.assertTrue(CondBernoulli.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli])]))

    # Bernoulli feature type instance
    tc.assertTrue(CondBernoulli.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli(0.5)])]))

    # invalid feature type
    tc.assertFalse(CondBernoulli.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # non-conditional scope
    tc.assertFalse(CondBernoulli.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        CondBernoulli.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondBernoulli.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])])
    CondBernoulli.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli])])
    CondBernoulli.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli(p=0.75)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondBernoulli.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondBernoulli.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondBernoulli.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondBernoulliInst = CondBernoulliBase
    elif tl.get_backend() == "pytorch":
        CondBernoulliInst = CondBernoulliTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondBernoulli))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondBernoulli,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    bernoulli = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Bernoulli])])
    tc.assertTrue(isinstance(bernoulli, CondBernoulliInst))

def test_structural_marginalization(do_for_all_backends):

    bernoulli = CondBernoulli(Scope([0], [1]))

    tc.assertTrue(marginalize(bernoulli, [1]) is not None)
    tc.assertTrue(marginalize(bernoulli, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]

    cond_bernoulli = CondBernoulli(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_bernoulli_updated = updateBackend(cond_bernoulli)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_bernoulli.scopes_out == cond_bernoulli_updated.scopes_out))

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    p = random.random()
    model_default = CondBernoulli(Scope([0], [1]))
    model_default.set_cond_f(lambda data: {"p": p})
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(param, float))
    else:
        tc.assertTrue(param.dtype == tl.float32)

    # change to float64 model
    model_updated = CondBernoulli(Scope([0], [1]))
    model_updated.set_cond_f(lambda data: {"p": p})
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
    model_default = CondBernoulli(Scope([0], [1]))
    model_default.set_cond_f(lambda data: {"p": p})
    model_updated = CondBernoulli(Scope([0], [1]))
    model_updated.set_cond_f(lambda data: {"p": p})
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
