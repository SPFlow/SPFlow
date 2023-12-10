import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_gamma import CondGamma as CondGammaBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gamma import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gamma import CondGamma as CondGammaTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_gamma import CondGamma
from spflow.torch.structure.spn.nodes.sum_node import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    gamma = CondGamma(Scope([0], [1]))
    tc.assertTrue(gamma.cond_f is None)
    gamma = CondGamma(Scope([0], [1]), lambda x: {"alpha": 1.0, "beta": 1.0})
    tc.assertTrue(isinstance(gamma.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondGamma, Scope([]))
    tc.assertRaises(Exception, CondGamma, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondGamma, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Gamma distribution: alpha>0, beta>0

    gamma = CondGamma(Scope([0], [1]))

    # alpha > 0
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
            "beta": tl.tensor(1.0, dtype=tl.float32),
        }
    )
    tc.assertTrue(
        gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[0]
        == tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32))
    )
    # alpha = 0
    gamma.set_cond_f(lambda data: {"alpha": tl.tensor(0.0, dtype=tl.float32), "beta": tl.tensor(1.0, dtype=tl.float32)})
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # alpha < 0
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0), -tl.tensor(1.0)),
            "beta": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # alpha = inf and alpha = nan
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(float("inf")),
            "beta": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(float("nan")),
            "beta": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # beta > 0
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(1.0, dtype=tl.float32),
            "beta": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
        }
    )
    tc.assertTrue(
        gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[1]
        == tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32))
    )
    # beta = 0
    gamma.set_cond_f(lambda data: {"alpha": tl.tensor(1.0, dtype=tl.float32), "beta": tl.tensor(0.0, dtype=tl.float32)})
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # beta < 0
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(0.0, dtype=tl.float32),
            "beta": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), -tl.tensor(1.0, dtype=tl.float32)),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # beta = inf and beta = non
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(1.0),
            "beta": tl.tensor(float("inf")),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(1.0),
            "beta": tl.tensor(float("nan")),
        }
    )
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(CondGamma.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # Gamma feature type class
    tc.assertTrue(CondGamma.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]))

    # Gamma feature type instance
    tc.assertTrue(CondGamma.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma(1.0, 1.0)])]))

    # invalid feature type
    tc.assertFalse(CondGamma.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # non-conditional scope
    tc.assertFalse(CondGamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondGamma.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondGamma.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])])
    CondGamma.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])])
    CondGamma.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma(1.5, 0.5)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGamma.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGamma.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGamma.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondGammaInst = CondGammaBase
    elif tl.get_backend() == "pytorch":
        CondGammaInst = CondGammaTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGamma))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondGamma,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gamma = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])])
    tc.assertTrue(isinstance(gamma, CondGammaInst))

def test_structural_marginalization(do_for_all_backends):

    gamma = CondGamma(Scope([0], [1]))

    tc.assertTrue(marginalize(gamma, [1]) is not None)
    tc.assertTrue(marginalize(gamma, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_gamma = CondGamma(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_gamma_updated = updateBackend(cond_gamma)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_gamma.scopes_out == cond_gamma_updated.scopes_out))

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    model_default = CondGamma(Scope([0], [1]))
    model_default.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
            "beta": tl.tensor(1.0, dtype=tl.float32),
        }
    )
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    param1 = param[0]
    param2 = param[1]

    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        if (isinstance(param1, np.ndarray)):
                tc.assertTrue(param1.dtype==tl.float32)
                tc.assertTrue(param2.dtype == tl.float32)
        else:
            tc.assertTrue(isinstance(param1, float))
            tc.assertTrue(isinstance(param2, float))
    else:
        tc.assertTrue(param1.dtype == tl.float32)
        tc.assertTrue(param2.dtype == tl.float32)

    # change to float64 model
    model_updated = CondGamma(Scope([0], [1]))
    model_updated.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
            "beta": tl.tensor(1.0, dtype=tl.float32),
        }
    )
    model_updated.to_dtype(tl.float64)
    param_up = model_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    param_up1 = param_up[0]
    param_up2 = param_up[1]
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        if (isinstance(param_up1, np.ndarray)):
            tc.assertTrue(param_up1.dtype == tl.float64)
            tc.assertTrue(param_up2.dtype == tl.float64)
        else:
            tc.assertTrue(isinstance(param_up1, float))
            tc.assertTrue(isinstance(param_up2, float))
    else:
        tc.assertTrue(param_up1.dtype == tl.float64)
        tc.assertTrue(param_up2.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([param]),
            np.array([param_up]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    model_default = CondGamma(Scope([0], [1]))
    model_default.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
            "beta": tl.tensor(1.0, dtype=tl.float32),
        }
    )
    model_updated = CondGamma(Scope([0], [1]))
    model_updated.set_cond_f(
        lambda data: {
            "alpha": tl_nextafter(tl.tensor(0.0, dtype=tl.float32), tl.tensor(1.0, dtype=tl.float32)),
            "beta": tl.tensor(1.0, dtype=tl.float32),
        }
    )
    # put model on gpu
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    model_updated.to_device(cuda)
    param = model_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    param1 = param[0]
    param2 = param[1]
    param_up = model_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    param_up1 = param_up[0]
    param_up2 = param_up[1]



    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(param1.device.type == "cpu")
    tc.assertTrue(param_up1.device.type == "cuda")

    tc.assertTrue(param2.device.type == "cpu")
    tc.assertTrue(param_up2.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([param1]),
            np.array([param_up1.cpu()]),
        )
    )

    tc.assertTrue(
        np.allclose(
            np.array([param2]),
            np.array([param_up2.cpu()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
