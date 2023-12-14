import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.node.leaf.hypergeometric import updateBackend
from spflow.base.structure.general.node.leaf.hypergeometric import Hypergeometric as HypergeometricBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.general.node.leaf.hypergeometric import Hypergeometric as HypergeometricTorch
from spflow.tensorly.structure.general.node.leaf.general_hypergeometric import Hypergeometric
from spflow.torch.structure import marginalize
from spflow.tensorly.utils.helper_functions import tl_nextafter, tl_toNumpy

tc = unittest.TestCase()

def test_initialization(do_for_all_backends):

    # Valid parameters for Hypergeometric distribution: N in N U {0}, M in {0,...,N}, n in {0,...,N}, p in [0,1]

    # N = 0
    Hypergeometric(Scope([0]), 0, 0, 0)
    # N < 0
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), -1, 1, 1)
    # N = inf and N = nan
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), np.inf, 1, 1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), np.nan, 1, 1)
    # N float
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1.5, 1, 1)

    # M < 0 and M > N
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, -1, 1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 2, 1)
    # 0 <= M <= N
    for i in range(4):
        Hypergeometric(Scope([0]), 3, i, 0)
    # M = inf and M = nan
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.inf, 1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.nan, 1)
    # M float
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 0.5, 1)

    # n < 0 and n > N
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, -1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 2)
    # 0 <= n <= N
    for i in range(4):
        Hypergeometric(Scope([0]), 3, 0, i)
    # n = inf and n = nan
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.inf)
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.nan)
    # n float
    tc.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 0.5)

    # invalid scopes
    tc.assertRaises(Exception, Hypergeometric, Scope([]), 1, 1, 1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0, 1]), 1, 1, 1)
    tc.assertRaises(Exception, Hypergeometric, Scope([0], [1]), 1, 1, 1)

def test_structural_marginalization(do_for_all_backends):

    hypergeometric = Hypergeometric(Scope([0]), 0, 0, 0)

    tc.assertTrue(marginalize(hypergeometric, [1]) is not None)
    tc.assertTrue(marginalize(hypergeometric, [0]) is None)

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # Bernoulli feature type instance
    tc.assertTrue(
        Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])])
    )

    # invalid feature type
    tc.assertFalse(Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # conditional scope
    tc.assertFalse(
        Hypergeometric.accepts(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.Hypergeometric(N=4, M=2, n=3)],
                )
            ]
        )
    )

    # multivariate signature
    tc.assertFalse(
        Hypergeometric.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                        FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    hypergeometric = Hypergeometric.from_signatures(
        [FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])]
    )
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.N), tl.tensor(4)))
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.M), tl.tensor(2)))
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.n), tl.tensor(3)))

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        Hypergeometric.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        Hypergeometric.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        Hypergeometric.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Hypergeometric.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        HypergeometricInst = HypergeometricBase
    elif tl.get_backend() == "pytorch":
        HypergeometricInst = HypergeometricTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(Hypergeometric))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        Hypergeometric,
        AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])]),
    )

    # make sure AutoLeaf can return correctly instantiated object
    hypergeometric = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])])
    tc.assertTrue(isinstance(hypergeometric, HypergeometricInst))
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.N), tl.tensor(4)))
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.M), tl.tensor(2)))
    tc.assertTrue(np.isclose(tl_toNumpy(hypergeometric.n), tl.tensor(3)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    N = 15
    M = 10
    n = 10
    hypergeometric = Hypergeometric(Scope([0]), N, M, n)
    for backend in backends:
        with tl.backend_context(backend):
            hypergeometric_updated = updateBackend(hypergeometric)

            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*hypergeometric.get_params()]),
                    np.array([*hypergeometric_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
