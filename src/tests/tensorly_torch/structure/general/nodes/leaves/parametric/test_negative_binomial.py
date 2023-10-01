import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as NegativeBinomialBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as NegativeBinomialTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_negative_binomial import NegativeBinomial
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Negative Binomial distribution: p in (0,1], n > 0

    # p = 1
    negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
    # p = 0
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, 0.0)
    # p < 0 and p > 1
    tc.assertRaises(
        Exception,
        NegativeBinomial,
        Scope([0]),
        1,
        tl_nextafter(tl.tensor(1.0), tl.tensor(2.0)),
    )
    tc.assertRaises(
        Exception,
        NegativeBinomial,
        [0],
        1,
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
    )
    # p = +-inf and p = nan
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.inf)
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, -np.inf)
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nan)

    # n = 0
    NegativeBinomial(Scope([0]), 0.0, 1.0)
    # n < 0
    tc.assertRaises(
        Exception,
        NegativeBinomial,
        Scope([0]),
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
        1.0,
    )
    # n = inf and n = nan
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), np.inf, 1.0)
    tc.assertRaises(Exception, NegativeBinomial, Scope([0]), np.nan, 1.0)

    # invalid scopes
    tc.assertRaises(Exception, NegativeBinomial, Scope([]), 1, 1.0)
    tc.assertRaises(Exception, NegativeBinomial, Scope([0, 1]), 1, 1.0)
    tc.assertRaises(Exception, NegativeBinomial, Scope([0], [1]), 1, 1.0)

def test_structural_marginalization(do_for_all_backends):

    negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

    tc.assertTrue(marginalize(negative_binomial, [1]) is not None)
    tc.assertTrue(marginalize(negative_binomial, [0]) is None)

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(NegativeBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type instance
    tc.assertTrue(NegativeBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])]))

    # invalid feature type
    tc.assertFalse(NegativeBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(
        NegativeBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)])])
    )

    # multivariate signature
    tc.assertFalse(
        NegativeBinomial.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.NegativeBinomial(n=3),
                        FeatureTypes.Binomial(n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    negative_binomial = NegativeBinomial.from_signatures(
        [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])]
    )
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.p), tl.tensor(0.5)))

    negative_binomial = NegativeBinomial.from_signatures(
        [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)])]
    )
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.p), tl.tensor(0.75)))

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        NegativeBinomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        NegativeBinomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        NegativeBinomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        NegativeBinomial.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        NegativeBinomialInst = NegativeBinomialBase
    elif tl.get_backend() == "pytorch":
        NegativeBinomialInst = NegativeBinomialTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(NegativeBinomial))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        NegativeBinomial,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    negative_binomial = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)])])
    tc.assertTrue(isinstance(negative_binomial, NegativeBinomialInst))
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(negative_binomial.p), tl.tensor(0.75)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)
    p = random.random()
    negative_binomial = NegativeBinomial(Scope([0]), n, p)
    for backend in backends:
        with tl.backend_context(backend):
            negative_binomial_updated = updateBackend(negative_binomial)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*negative_binomial.get_params()]),
                    np.array([*negative_binomial_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
