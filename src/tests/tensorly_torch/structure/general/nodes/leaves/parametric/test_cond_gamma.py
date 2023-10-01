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
            "alpha": tl_nextafter(tl.tensor(0.0), tl.tensor(1.0)),
            "beta": tl.tensor(1.0),
        }
    )
    tc.assertTrue(
        gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[0]
        == tl_nextafter(tl.tensor(0.0), tl.tensor(1.0))
    )
    # alpha = 0
    gamma.set_cond_f(lambda data: {"alpha": tl.tensor(0.0), "beta": tl.tensor(1.0)})
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
            "alpha": tl.tensor(1.0),
            "beta": tl_nextafter(tl.tensor(0.0), tl.tensor(1.0)),
        }
    )
    tc.assertTrue(
        gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[1]
        == tl_nextafter(tl.tensor(0.0), tl.tensor(1.0))
    )
    # beta = 0
    gamma.set_cond_f(lambda data: {"alpha": tl.tensor(1.0), "beta": tl.tensor(0.0)})
    tc.assertRaises(
        Exception,
        gamma.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # beta < 0
    gamma.set_cond_f(
        lambda data: {
            "alpha": tl.tensor(0.0),
            "beta": tl_nextafter(tl.tensor(0.0), -tl.tensor(1.0)),
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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
