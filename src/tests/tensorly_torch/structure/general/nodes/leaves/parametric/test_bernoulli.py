import random
import unittest

import numpy as np
import torch
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy
from spflow.base.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as BernoulliBase
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as BernoulliTorch
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_bernoulli import Bernoulli
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import updateBackend
from spflow.tensorly.structure import AutoLeaf

tc = unittest.TestCase()


def test_initialization(do_for_all_backends):

    # Valid parameters for Bernoulli distribution: p in [0,1]

    # p = 0
    bernoulli = Bernoulli(Scope([0]), 0.0)
    # p = 1
    bernoulli = Bernoulli(Scope([0]), 1.0)
    # p < 0 and p > 1
    tc.assertRaises(
        Exception,
        Bernoulli,
        Scope([0]),
        tl_nextafter(tl.tensor(1.0), tl.tensor(2.0)),
    )
    tc.assertRaises(
        Exception,
        Bernoulli,
        Scope([0]),
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
    )
    # p = inf and p = nan
    tc.assertRaises(Exception, Bernoulli, Scope([0]), np.inf)
    tc.assertRaises(Exception, Bernoulli, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Bernoulli, Scope([]), 0.5)
    tc.assertRaises(Exception, Bernoulli, Scope([0, 1]), 0.5)
    tc.assertRaises(Exception, Bernoulli, Scope([0], [1]), 0.5)

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(Bernoulli.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type class
    tc.assertTrue(Bernoulli.accepts([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]))

    # Bernoulli feature type instance
    tc.assertTrue(Bernoulli.accepts([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.5)])]))

    # invalid feature type
    tc.assertFalse(Bernoulli.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(Bernoulli.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        Bernoulli.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    bernoulli = Bernoulli.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
    tc.assertTrue(np.isclose(tl_toNumpy(bernoulli.p), tl.tensor(0.5)))

    bernoulli = Bernoulli.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])])
    tc.assertTrue(np.isclose(tl_toNumpy(bernoulli.p), tl.tensor(0.5)))

    bernoulli = Bernoulli.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])])
    tc.assertTrue(np.isclose(tl_toNumpy(bernoulli.p), tl.tensor(0.75)))

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Bernoulli.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Bernoulli.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Bernoulli.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        BernoulliInst = BernoulliBase
    elif tl.get_backend() == "pytorch":
        BernoulliInst = BernoulliTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Bernoulli))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Bernoulli,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    bernoulli = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])])
    tc.assertTrue(isinstance(bernoulli, BernoulliInst))
    tc.assertTrue(np.isclose(tl_toNumpy(bernoulli.p), tl.tensor(0.75)))

def test_structural_marginalization(do_for_all_backends):

    bernoulli = Bernoulli(Scope([0]), 0.5)

    tc.assertTrue(marginalize(bernoulli, [1]) is not None)
    tc.assertTrue(marginalize(bernoulli, [0]) is None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    p = random.random()
    bernoulli = Bernoulli(Scope([0]), p)
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(bernoulli)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*bernoulli.get_params()]),
                    np.array([*bernoulli_updated.get_params()]),
                )
            )
        # check conversion from python to torch

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
