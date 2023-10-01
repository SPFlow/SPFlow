import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.uniform import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.uniform import Uniform as UniformBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.uniform import Uniform as UniformTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_uniform import Uniform
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

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
        tl_nextafter(tl.tensor(start_end), tl.tensor(-1.0)),
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

    uniform = Uniform.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)])])
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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
