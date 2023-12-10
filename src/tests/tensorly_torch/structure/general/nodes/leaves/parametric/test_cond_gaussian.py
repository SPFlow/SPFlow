import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gaussian import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_gaussian import CondGaussian
from spflow.torch.structure.spn.nodes.sum_node import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    gaussian = CondGaussian(Scope([0], [1]))
    tc.assertTrue(gaussian.cond_f is None)
    gaussian = CondGaussian(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
    tc.assertTrue(isinstance(gaussian.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondGaussian, Scope([]))
    tc.assertRaises(Exception, CondGaussian, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondGaussian, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Gaussian distribution: mean in R, std > 0

    gaussian = CondGaussian(Scope([0], [1]))

    # mean = inf and mean = nan
    gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor(float("inf")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gaussian.set_cond_f(
        lambda data: {
            "mean": -tl.tensor(float("inf")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor(float("nan")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # std = 0 and std < 0
    gaussian.set_cond_f(lambda data: {"mean": tl.tensor(0.0), "std": tl.tensor(0.0)})
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # std = inf and std = nan
    gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": -tl.tensor(float("inf")),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    gaussian.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": tl.tensor(float("nan")),
        }
    )
    tc.assertRaises(
        Exception,
        gaussian.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(CondGaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # Gaussian feature type class
    tc.assertTrue(CondGaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]))

    # Gaussian feature type instance
    tc.assertTrue(CondGaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian(0.0, 1.0)])]))

    # invalid feature type
    tc.assertFalse(CondGaussian.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # non-conditional scope
    tc.assertFalse(CondGaussian.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondGaussian.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondGaussian.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])])
    CondGaussian.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])])
    CondGaussian.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian(-1.0, 1.5)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGaussian.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGaussian.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGaussian.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondGaussianInst = CondGaussianBase
    elif tl.get_backend() == "pytorch":
        CondGaussianInst = CondGaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGaussian))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondGaussian,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gaussian = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])])
    tc.assertTrue(isinstance(gaussian, CondGaussianInst))

def test_structural_marginalization(do_for_all_backends):

    gaussian = CondGaussian(Scope([0], [1]))

    tc.assertTrue(marginalize(gaussian, [1]) is not None)
    tc.assertTrue(marginalize(gaussian, [0]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_gaussian = CondGaussian(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_gaussian_updated = updateBackend(cond_gaussian)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_gaussian.scopes_out == cond_gaussian_updated.scopes_out))

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    model_default = CondGaussian(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
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
    model_updated = CondGaussian(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
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
    model_default = CondGaussian(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
    model_updated = CondGaussian(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
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
