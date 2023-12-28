import random
import unittest

import numpy as np
import torch
import tensorly as tl
from spflow.utils import Tensor
from spflow.tensor import ops as tle
from spflow.base.structure.general.node.leaf.bernoulli import Bernoulli as BernoulliBase
from spflow.torch.structure.general.node.leaf.bernoulli import Bernoulli as BernoulliTorch
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.structure.general.node.leaf.general_bernoulli import Bernoulli
from spflow.torch.structure.general.node.leaf.bernoulli import updateBackend
from spflow.structure import AutoLeaf

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
        tle.nextafter(tl.tensor(1.0), tl.tensor(2.0)),
    )
    tc.assertRaises(
        Exception,
        Bernoulli,
        Scope([0]),
        tle.nextafter(tl.tensor(0.0), tl.tensor(-1.0)),
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
    tc.assertTrue(np.isclose(tle.toNumpy(bernoulli.p), tl.tensor(0.5)))

    bernoulli = Bernoulli.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])])
    tc.assertTrue(np.isclose(tle.toNumpy(bernoulli.p), tl.tensor(0.5)))

    bernoulli = Bernoulli.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])])
    tc.assertTrue(np.isclose(tle.toNumpy(bernoulli.p), tl.tensor(0.75)))

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
    tc.assertTrue(np.isclose(tle.toNumpy(bernoulli.p), tl.tensor(0.75)))


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


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    p = random.random()
    model_default = Bernoulli(Scope([0]), p)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.p, float))
    else:
        tc.assertTrue(model_default.p.dtype == tl.float32)

    # change to float64 model
    model_updated = Bernoulli(Scope([0]), p)
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
    p = random.random()
    torch.set_default_dtype(torch.float32)
    model_default = Bernoulli(Scope([0]), p)
    model_updated = Bernoulli(Scope([0]), p)
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
