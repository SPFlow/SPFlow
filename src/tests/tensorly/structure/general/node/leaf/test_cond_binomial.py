import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.node.leaf.cond_binomial import CondBinomial as CondBinomialBase
from spflow.torch.structure.general.node.leaf.cond_binomial import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.node.leaf.cond_binomial import CondBinomial as CondBinomialTorch
from spflow.tensorly.structure.general.node.leaf.general_cond_binomial import CondBinomial

from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    binomial = CondBinomial(Scope([0], [1]), n=1)
    tc.assertTrue(binomial.cond_f is None)
    binomial = CondBinomial(Scope([0], [1]), n=1, cond_f=lambda x: {"p": 0.5})
    tc.assertTrue(isinstance(binomial.cond_f, Callable))

    # n = 0
    binomial = CondBinomial(Scope([0], [1]), 0, 0.5)
    # n < 0
    tc.assertRaises(Exception, CondBinomial, Scope([0]), -1)
    # n float
    tc.assertRaises(Exception, CondBinomial, Scope([0]), 0.5)
    # n = inf and n = nan
    tc.assertRaises(Exception, CondBinomial, Scope([0]), np.inf)
    tc.assertRaises(Exception, CondBinomial, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, CondBinomial, Scope([]), 1)
    tc.assertRaises(Exception, CondBinomial, Scope([0, 1], [2]), 1)
    tc.assertRaises(Exception, CondBinomial, Scope([0]), 1)

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

    binomial = CondBinomial(Scope([0], [1]), n=1)

    # p = 0
    binomial.set_cond_f(lambda data: {"p": 0.0})
    tc.assertTrue(binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 0.0)
    # p = 1
    binomial.set_cond_f(lambda data: {"p": 1.0})
    tc.assertTrue(binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 1.0)
    # p < 0 and p > 1
    binomial.set_cond_f(lambda data: {"p": tl_nextafter(tl.tensor(1.0), tl.tensor(2.0))})
    tc.assertRaises(
        Exception,
        binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    binomial.set_cond_f(lambda data: {"p": tl_nextafter(tl.tensor(0.0), -tl.tensor(1.0))})
    tc.assertRaises(
        Exception,
        binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # p = inf and p = nan
    binomial.set_cond_f(lambda data: {"p": tl.tensor(float("inf"))})
    tc.assertRaises(
        Exception,
        binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    binomial.set_cond_f(lambda data: {"p": tl.tensor(float("nan"))})
    tc.assertRaises(
        Exception,
        binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # feature type instance
    tc.assertTrue(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]))

    # invalid feature type
    tc.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # non-conditional scope
    tc.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])]))

    # multivariate signature
    tc.assertFalse(
        CondBinomial.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Binomial(n=3),
                        FeatureTypes.Binomial(n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])])
    CondBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3, p=0.75)])])

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        CondBinomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondBinomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        CondBinomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondBinomial.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondBinomialInst = CondBinomialBase
    elif tl.get_backend() == "pytorch":
        CondBinomialInst = CondBinomialTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")
    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondBinomial))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondBinomial,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    binomial = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])])
    tc.assertTrue(isinstance(binomial, CondBinomialInst))
    tc.assertEqual(binomial.n, 3)

def test_structural_marginalization(do_for_all_backends):

    binomial = CondBinomial(Scope([0], [1]), 1)

    tc.assertTrue(marginalize(binomial, [1]) is not None)
    tc.assertTrue(marginalize(binomial, [0]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)
    cond_binomial = CondBinomial(Scope([0], [1]), n)
    for backend in backends:
        with tl.backend_context(backend):
            cond_binomial_updated = updateBackend(cond_binomial)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_binomial.scopes_out == cond_binomial_updated.scopes_out))

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    p = random.random()
    n = random.randint(2, 10)
    model_default = CondBinomial(Scope([0], [1]), n)
    model_default.set_cond_f(lambda data: {"p": p})
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(param, float))
    else:
        tc.assertTrue(param.dtype == tl.float32)

    # change to float64 model
    model_updated = CondBinomial(Scope([0], [1]), n)
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
    n = random.randint(2, 10)
    torch.set_default_dtype(torch.float32)
    model_default = CondBinomial(Scope([0], [1]), n)
    model_default.set_cond_f(lambda data: {"p": p})
    model_updated = CondBinomial(Scope([0], [1]), n)
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
