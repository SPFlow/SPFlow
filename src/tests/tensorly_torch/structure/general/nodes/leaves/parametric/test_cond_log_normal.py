import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_log_normal import CondLogNormal as CondLogNormalBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_log_normal import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_log_normal import CondLogNormal as CondLogNormalTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_log_normal import CondLogNormal
from spflow.tensorly.structure.spn.nodes.sum_node import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    log_normal = CondLogNormal(Scope([0], [1]))
    tc.assertTrue(log_normal.cond_f is None)
    log_normal = CondLogNormal(Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0})
    tc.assertTrue(isinstance(log_normal.cond_f, Callable))

    # invalid scopes
    tc.assertRaises(Exception, CondLogNormal, Scope([]))
    tc.assertRaises(Exception, CondLogNormal, Scope([0, 1], [2]))
    tc.assertRaises(Exception, CondLogNormal, Scope([0]))

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Log-Normal distribution: mean in (-inf,inf), std in (0,inf)

    log_normal = CondLogNormal(Scope([0], [1]))

    # mean = +-inf and mean = 0
    log_normal.set_cond_f(
        lambda data: {
            "mean": tl.tensor(float("inf")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    log_normal.set_cond_f(
        lambda data: {
            "mean": -tl.tensor(float("inf")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    log_normal.set_cond_f(
        lambda data: {
            "mean": tl.tensor(float("nan")),
            "std": tl.tensor(1.0),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # std <= 0
    log_normal.set_cond_f(lambda data: {"mean": tl.tensor(0.0), "std": tl.tensor(0.0)})
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    log_normal.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # std = +-inf and std = nan
    log_normal.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": -tl.tensor(float("inf")),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    log_normal.set_cond_f(
        lambda data: {
            "mean": tl.tensor(0.0),
            "std": tl.tensor(float("nan")),
        }
    )
    tc.assertRaises(
        Exception,
        log_normal.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(CondLogNormal.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # LogNormal feature type class
    tc.assertTrue(CondLogNormal.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]))

    # LogNormal feature type instance
    tc.assertTrue(CondLogNormal.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal(0.0, 1.0)])]))

    # invalid feature type
    tc.assertFalse(CondLogNormal.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # non-conditional scope
    tc.assertFalse(CondLogNormal.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondLogNormal.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondLogNormal.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])])
    CondLogNormal.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])])
    CondLogNormal.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal(-1.0, 1.5)])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondLogNormal.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondLogNormal.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondLogNormal.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondLogNormalInst = CondLogNormalBase
    elif tl.get_backend() == "pytorch":
        CondLogNormalInst = CondLogNormalTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")


    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondLogNormal))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondLogNormal,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    log_normal = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])])
    tc.assertTrue(isinstance(log_normal, CondLogNormalInst))

def test_structural_marginalization(do_for_all_backends):

    log_normal = CondLogNormal(Scope([0], [1]))

    tc.assertTrue(marginalize(log_normal, [1]) is not None)
    tc.assertTrue(marginalize(log_normal, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_log_normal = CondLogNormal(Scope([0], [1]))
    for backend in backends:
        with tl.backend_context(backend):
            cond_log_normal_updated = updateBackend(cond_log_normal)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_log_normal.scopes_out == cond_log_normal_updated.scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
