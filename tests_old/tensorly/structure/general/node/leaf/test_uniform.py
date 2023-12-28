import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.node.leaf.uniform import updateBackend
from spflow.base.structure.general.node.leaf.uniform import Uniform as UniformBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.autoleaf import AutoLeaf
from spflow.torch.structure.general.node.leaf.uniform import Uniform as UniformTorch
from spflow.modules.node import Uniform
from spflow.torch.structure import marginalize
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_initialization(do_for_all_backends):
    # Valid parameters for Uniform distribution: a<b

    # start = end
    start_end = random.random()
    tc.assertRaises(Exception, Uniform, Scope([0]), start_end, start_end)
    # start > end
    tc.assertRaises(
        Exception,
        Uniform,
        Scope([0]),
        start_end,
        tle.nextafter(tl.tensor(start_end), tl.tensor(-1.0)),
    )
    # start = +-inf and start = nan
    tc.assertRaises(Exception, Uniform, Scope([0]), np.inf, 0.0)
    tc.assertRaises(Exception, Uniform, Scope([0]), -np.inf, 0.0)
    tc.assertRaises(Exception, Uniform, Scope([0]), np.nan, 0.0)
    # end = +-inf and end = nan
    tc.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.inf)
    tc.assertRaises(Exception, Uniform, Scope([0]), 0.0, -np.inf)
    tc.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Uniform, Scope([]), 0.0, 1.0)
    tc.assertRaises(Exception, Uniform, Scope([0, 1]), 0.0, 1.0)
    tc.assertRaises(Exception, Uniform, Scope([0], [1]), 0.0, 1.0)


def test_structural_marginalization(do_for_all_backends):
    uniform = Uniform(Scope([0]), 0.0, 1.0)

    tc.assertTrue(marginalize(uniform, [1]) is not None)
    tc.assertTrue(marginalize(uniform, [0]) is None)


def test_accept(do_for_all_backends):
    # discrete meta type (should reject)
    tc.assertFalse(Uniform.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # feature type instance
    tc.assertTrue(Uniform.accepts([FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)])]))

    # invalid feature type
    tc.assertFalse(Uniform.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # conditional scope
    tc.assertFalse(
        Uniform.accepts(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.Uniform(start=-1.0, end=2.0)],
                )
            ]
        )
    )

    # multivariate signature
    tc.assertFalse(
        Uniform.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Uniform(start=-1.0, end=2.0),
                        FeatureTypes.Uniform(start=-1.0, end=2.0),
                    ],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    uniform = Uniform.from_signatures(
        [FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)])]
    )
    tc.assertTrue(np.isclose(uniform.start, tl.tensor(-1.0)))
    tc.assertTrue(np.isclose(uniform.end, tl.tensor(2.0)))

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        Uniform.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Uniform.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Uniform.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Uniform.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    if tl.get_backend() == "numpy":
        UniformInst = UniformBase
    elif tl.get_backend() == "pytorch":
        UniformInst = UniformTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Uniform))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Uniform,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    uniform = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)])])
    tc.assertTrue(isinstance(uniform, UniformInst))
    tc.assertTrue(np.isclose(uniform.start, tl.tensor(-1.0)))
    tc.assertTrue(np.isclose(uniform.end, tl.tensor(2.0)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    start = random.random()
    end = start + 1e-7 + random.random()
    uniform = Uniform(Scope([0]), start, end)
    for backend in backends:
        with tl.backend_context(backend):
            uniform_updated = updateBackend(uniform)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*uniform.get_params()]),
                    np.array([*uniform_updated.get_params()]),
                )
            )


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    start = random.random()
    end = start + 1e-7 + random.random()
    model_default = Uniform(Scope([0]), start, end)
    tc.assertTrue(model_default.dtype == tl.float32)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_default.start, float))
        tc.assertTrue(isinstance(model_default.end, float))
    else:
        tc.assertTrue(model_default.start.dtype == tl.float32)
        tc.assertTrue(model_default.end.dtype == tl.float32)

    # change to float64 model
    model_updated = Uniform(Scope([0]), start, end)
    model_updated.to_dtype(tl.float64)
    tc.assertTrue(model_updated.dtype == tl.float64)
    if do_for_all_backends == "numpy":
        tc.assertTrue(isinstance(model_updated.start, float))
        tc.assertTrue(isinstance(model_updated.end, float))
    else:
        tc.assertTrue(model_updated.start.dtype == tl.float64)
        tc.assertTrue(model_updated.end.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    start = random.random()
    end = start + 1e-7 + random.random()
    torch.set_default_dtype(torch.float32)
    model_default = Uniform(Scope([0]), start, end)
    model_updated = Uniform(Scope([0]), start, end)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    tc.assertTrue(model_default.device.type == "cpu")
    tc.assertTrue(model_updated.device.type == "cuda")

    tc.assertTrue(model_default.start.device.type == "cpu")
    tc.assertTrue(model_updated.start.device.type == "cuda")

    tc.assertTrue(model_default.end.device.type == "cpu")
    tc.assertTrue(model_updated.end.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*model_default.get_params()]),
            np.array([*model_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
