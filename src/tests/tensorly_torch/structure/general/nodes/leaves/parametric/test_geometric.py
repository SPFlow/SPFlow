import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.geometric import updateBackend
from spflow.base.structure.general.nodes.leaves.parametric.geometric import Geometric as GeometricBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.nodes.leaves.parametric.geometric import Geometric as GeometricTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_geometric import Geometric
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialiation(do_for_all_backends):

    # Valid parameters for Geometric distribution: p in (0,1]

    # p = 0
    tc.assertRaises(Exception, Geometric, Scope([0]), 0.0)

    # p = inf and p = nan
    tc.assertRaises(Exception, Geometric, Scope([0]), np.inf)
    tc.assertRaises(Exception, Geometric, Scope([0]), np.nan)

    # invalid scopes
    tc.assertRaises(Exception, Geometric, Scope([]), 0.5)
    tc.assertRaises(Exception, Geometric, Scope([0, 1]), 0.5)
    tc.assertRaises(Exception, Geometric, Scope([0], [1]), 0.5)

def test_structural_marginalization(do_for_all_backends):

    geometric = Geometric(Scope([0]), 0.5)

    tc.assertTrue(marginalize(geometric, [1]) is not None)
    tc.assertTrue(marginalize(geometric, [0]) is None)

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(Geometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Geometric feature type class
    tc.assertTrue(Geometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Geometric])]))

    # Geometric feature type instance
    tc.assertTrue(Geometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)])]))

    # invalid feature type
    tc.assertFalse(Geometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(Geometric.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        Geometric.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    geometric = Geometric.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
    tc.assertTrue(np.isclose(tl_toNumpy(geometric.p), tl.tensor(0.5)))

    geometric = Geometric.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Geometric])])
    tc.assertTrue(np.isclose(tl_toNumpy(geometric.p), tl.tensor(0.5)))

    geometric = Geometric.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])])
    tc.assertTrue(np.isclose(tl_toNumpy(geometric.p), tl.tensor(0.75)))

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Geometric.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Geometric.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Geometric.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        GeometricInst = GeometricBase
    elif tl.get_backend() == "pytorch":
        GeometricInst = GeometricTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Geometric))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Geometric,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Geometric])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    geometric = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])])
    tc.assertTrue(isinstance(geometric, GeometricInst))
    tc.assertTrue(np.isclose(tl_toNumpy(geometric.p), tl.tensor(0.75)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    p = random.random()
    geometric = Geometric(Scope([0]), p)
    for backend in backends:
        with tl.backend_context(backend):
            geometric_updated = updateBackend(geometric)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*geometric.get_params()]),
                    np.array([*geometric_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
