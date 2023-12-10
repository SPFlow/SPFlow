import random
import unittest

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.binomial import Binomial as BinomialBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.binomial import Binomial as BinomialTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_binomial import Binomial
from spflow.torch.structure.general.nodes.leaves.parametric.binomial import updateBackend
from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

    # p = 0
    binomial = Binomial(Scope([0]), 1, 0.0)
    # p = 1
    binomial = Binomial(Scope([0]), 1, 1.0)
    # p < 0 and p > 1
    tc.assertRaises(
        Exception,
        Binomial,
        Scope([0]),
        1,
        tl_nextafter(tl.tensor(1.0), tl.tensor(2.0)),
    )
    tc.assertRaises(
        Exception,
        Binomial,
        Scope([0]),
        1,
        tl_nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
    )
    # p = inf and p = nan
    tc.assertRaises(Exception, Binomial, Scope([0]), 1, np.inf)
    tc.assertRaises(Exception, Binomial, Scope([0]), 1, np.nan)

    # n = 0
    binomial = Binomial(Scope([0]), 0, 0.5)
    # n < 0
    tc.assertRaises(Exception, Binomial, Scope([0]), -1, 0.5)
    # n float
    tc.assertRaises(Exception, Binomial, Scope([0]), 0.5, 0.5)
    # n = inf and n = nan
    tc.assertRaises(Exception, Binomial, Scope([0]), np.inf, 0.5)
    tc.assertRaises(Exception, Binomial, Scope([0]), np.nan, 0.5)

    # invalid scopes
    tc.assertRaises(Exception, Binomial, Scope([]), 1, 0.5)
    tc.assertRaises(Exception, Binomial, Scope([0, 1]), 1, 0.5)
    tc.assertRaises(Exception, Binomial, Scope([0], [1]), 1, 0.5)

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(Binomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type instance
    tc.assertTrue(Binomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])]))

    # invalid feature type
    tc.assertFalse(Binomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(Binomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]))

    # multivariate signature
    tc.assertFalse(
        Binomial.accepts(
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

    binomial = Binomial.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])])
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.p), tl.tensor(0.5)))

    binomial = Binomial.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3, p=0.75)])])
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.p), tl.tensor(0.75)))

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        Binomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Binomial.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Binomial.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Binomial.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        BinomialInst = BinomialBase
    elif tl.get_backend() == "pytorch":
        BinomialInst = BinomialTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Binomial))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Binomial,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    binomial = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3, p=0.75)])])
    tc.assertTrue(isinstance(binomial, BinomialInst))
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.n), tl.tensor(3)))
    tc.assertTrue(np.isclose(tl_toNumpy(binomial.p), tl.tensor(0.75)))

def test_structural_marginalization(do_for_all_backends):

    binomial = Binomial(Scope([0]), 1, 0.5)

    tc.assertTrue(marginalize(binomial, [1]) is not None)
    tc.assertTrue(marginalize(binomial, [0]) is None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)
    p = random.random()
    binomial = Binomial(Scope([0]), n, p)
    for backend in backends:
        with tl.backend_context(backend):
            binomial_updated = updateBackend(binomial)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*binomial.get_params()]),
                    np.array([*binomial_updated.get_params()]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    n = random.randint(2, 10)
    p = random.random()
    model_default = Binomial(Scope([0]), n, p)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.p, float))
    else:
        tc.assertTrue(model_default.p.dtype == tl.float32)

    # change to float64 model
    model_updated = Binomial(Scope([0]), n, p)
    model_updated.to_dtype(tl.float64)
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_updated.p, float))
    else:
        tc.assertTrue(model_updated.p.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    n = random.randint(2, 10)
    p = random.random()
    torch.set_default_dtype(torch.float32)
    model_default = Binomial(Scope([0]), n, p)
    model_updated = Binomial(Scope([0]), n, p)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(model_default.p.device.type == "cpu")
    tc.assertTrue(model_updated.p.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
