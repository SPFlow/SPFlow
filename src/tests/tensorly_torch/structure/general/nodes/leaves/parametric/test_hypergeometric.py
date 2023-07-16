import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.torch.structure.general.nodes.leaves.parametric.hypergeometric import updateBackend
from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import Hypergeometric as BaseHypergeometric
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.spn import Hypergeometric as TorchHypergeometric
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_hypergeometric import Hypergeometric
from spflow.torch.structure import marginalize, toBase, toTorch
#from spflow.torch.structure.spn import Hypergeometric


class TestHypergeometric(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Hypergeometric distribution: N in N U {0}, M in {0,...,N}, n in {0,...,N}, p in [0,1]

        # N = 0
        Hypergeometric(Scope([0]), 0, 0, 0)
        # N < 0
        self.assertRaises(Exception, Hypergeometric, Scope([0]), -1, 1, 1)
        # N = inf and N = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), np.inf, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), np.nan, 1, 1)
        # N float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1.5, 1, 1)

        # M < 0 and M > N
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, -1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 2, 1)
        # 0 <= M <= N
        for i in range(4):
            Hypergeometric(Scope([0]), 3, i, 0)
        # M = inf and M = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.inf, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, np.nan, 1)
        # M float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 0.5, 1)

        # n < 0 and n > N
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, -1)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 2)
        # 0 <= n <= N
        for i in range(4):
            Hypergeometric(Scope([0]), 3, 0, i)
        # n = inf and n = nan
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.inf)
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, np.nan)
        # n float
        self.assertRaises(Exception, Hypergeometric, Scope([0]), 1, 1, 0.5)

        # invalid scopes
        self.assertRaises(Exception, Hypergeometric, Scope([]), 1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0, 1]), 1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, Scope([0], [1]), 1, 1, 1)

    def test_structural_marginalization(self):

        hypergeometric = Hypergeometric(Scope([0]), 0, 0, 0)

        self.assertTrue(marginalize(hypergeometric, [1]) is not None)
        self.assertTrue(marginalize(hypergeometric, [0]) is None)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # Bernoulli feature type instance
        self.assertTrue(
            Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])])
        )

        # invalid feature type
        self.assertFalse(Hypergeometric.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # conditional scope
        self.assertFalse(
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
        self.assertFalse(
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

    def test_initialization_from_signatures(self):

        hypergeometric = Hypergeometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])]
        )
        self.assertTrue(torch.isclose(hypergeometric.N, torch.tensor(4)))
        self.assertTrue(torch.isclose(hypergeometric.M, torch.tensor(2)))
        self.assertTrue(torch.isclose(hypergeometric.n, torch.tensor(3)))

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            Hypergeometric.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            Hypergeometric.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Hypergeometric.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Hypergeometric.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Hypergeometric))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Hypergeometric,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        hypergeometric = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)])])
        self.assertTrue(isinstance(hypergeometric, TorchHypergeometric))
        self.assertTrue(torch.isclose(hypergeometric.N, torch.tensor(4)))
        self.assertTrue(torch.isclose(hypergeometric.M, torch.tensor(2)))
        self.assertTrue(torch.isclose(hypergeometric.n, torch.tensor(3)))

    def test_base_backend_conversion(self):

        N = 15
        M = 10
        n = 10

        torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)
        node_hypergeometric = BaseHypergeometric(Scope([0]), N, M, n)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_hypergeometric.get_params()]),
                np.array([*toBase(torch_hypergeometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_hypergeometric.get_params()]),
                np.array([*toTorch(node_hypergeometric).get_params()]),
            )
        )

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        N = 15
        M = 10
        n = 10
        hypergeometric = Hypergeometric(Scope([0]), N, M, n)
        for backend in backends:
            tl.set_backend(backend)
            hypergeometric_updated = updateBackend(hypergeometric)

            # check conversion from torch to python
            self.assertTrue(
                np.allclose(
                    np.array([*hypergeometric.get_params()]),
                    np.array([*hypergeometric_updated.get_params()]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
