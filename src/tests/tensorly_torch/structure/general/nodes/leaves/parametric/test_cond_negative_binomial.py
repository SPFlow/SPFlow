import random
import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_negative_binomial import CondNegativeBinomial as CondNegativeBinomialBase
from spflow.torch.structure.general.nodes.leaves.parametric.cond_negative_binomial import updateBackend
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.cond_negative_binomial import CondNegativeBinomial as CondNegativeBinomialTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_negative_binomial import CondNegativeBinomial
from spflow.tensorly.structure.spn.nodes.sum_node import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    binomial = CondNegativeBinomial(Scope([0], [1]), n=1)
    tc.assertTrue(binomial.cond_f is None)
    binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=lambda x: {"p": 0.5})
    tc.assertTrue(isinstance(binomial.cond_f, Callable))

    # n = 0
    CondNegativeBinomial(Scope([0], [1]), 0.0)
    # n < 0
    tc.assertRaises(
        Exception,
        CondNegativeBinomial,
        Scope([0]),
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
    )
    # n = inf and n = nan
    tc.assertRaises(Exception, CondNegativeBinomial, Scope([0]), np.inf)
    tc.assertRaises(Exception, CondNegativeBinomial, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, CondNegativeBinomial, Scope([]), 1)
    tc.assertRaises(Exception, CondNegativeBinomial, Scope([0, 1], [2]), 1)
    tc.assertRaises(Exception, CondNegativeBinomial, Scope([0]), 1)

def test_retrieve_params(do_for_all_backends):

    # Valid parameters for Negative Binomial distribution: p in (0,1], n > 0
    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1)

    # p = 1
    negative_binomial.set_cond_f(lambda data: {"p": 1.0})
    tc.assertTrue(negative_binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 1.0)
    # p = 0
    negative_binomial.set_cond_f(lambda data: {"p": 0.0})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    # p < 0 and p > 1
    negative_binomial.set_cond_f(lambda data: {"p": tl_nextafter(tl.tensor(1.0), tl.tensor(2.0))})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    negative_binomial.set_cond_f(lambda data: {"p": tl_nextafter(tl.tensor(0.0), -tl.tensor(1.0))})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

    # p = +-inf and p = nan
    negative_binomial.set_cond_f(lambda data: {"p": tl.tensor(float("inf"))})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    negative_binomial.set_cond_f(lambda data: {"p": -tl.tensor(float("inf"))})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )
    negative_binomial.set_cond_f(lambda data: {"p": tl.tensor(float("nan"))})
    tc.assertRaises(
        Exception,
        negative_binomial.retrieve_params,
        np.array([[1.0]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type instance
    tc.assertTrue(
        CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])])
    )

    # invalid feature type
    tc.assertFalse(CondNegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # non-conditional scope
    tc.assertFalse(
        CondNegativeBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])])
    )

    # multivariate signature
    tc.assertFalse(
        CondNegativeBinomial.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.NegativeBinomial(n=3),
                        FeatureTypes.Binomial(n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    CondNegativeBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])])
    CondNegativeBinomial.from_signatures(
        [
            FeatureContext(
                Scope([0], [1]),
                [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
            )
        ]
    )

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        CondNegativeBinomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondNegativeBinomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondNegativeBinomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondNegativeBinomial.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondNegativeBinomialInst = CondNegativeBinomialBase
    elif tl.get_backend() == "pytorch":
        CondNegativeBinomialInst = CondNegativeBinomialTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondNegativeBinomial))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondNegativeBinomial,
        AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    negative_binomial = AutoLeaf(
        [
            FeatureContext(
                Scope([0], [1]),
                [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
            )
        ]
    )
    tc.assertTrue(isinstance(negative_binomial, CondNegativeBinomialInst))
    tc.assertEqual(negative_binomial.n, tl.tensor(3))

def test_structural_marginalization(do_for_all_backends):

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1)

    tc.assertTrue(marginalize(negative_binomial, [1]) is not None)
    tc.assertTrue(marginalize(negative_binomial, [0]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)

    cond_negative_binomial = CondNegativeBinomial(Scope([0], [1]), n)
    for backend in backends:
        with tl.backend_context(backend):
            cond_negative_binomial_updated = updateBackend(cond_negative_binomial)

            # check conversion from torch to python
            tc.assertTrue(np.all(cond_negative_binomial.scopes_out == cond_negative_binomial_updated.scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
