import unittest

import numpy as np

from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import Hypergeometric, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestHypergeometric(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Hypergeometric distribution: N in N U {0}, M in {0,...,N}, n in {0,...,N}

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

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            Hypergeometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            Hypergeometric.accepts(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            Hypergeometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

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
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]
                )
            ]
        )
        self.assertEqual(hypergeometric.N, 4)
        self.assertEqual(hypergeometric.M, 2)
        self.assertEqual(hypergeometric.n, 3)

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
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        hypergeometric = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]
                )
            ]
        )
        self.assertTrue(isinstance(hypergeometric, Hypergeometric))
        self.assertEqual(hypergeometric.N, 4)
        self.assertEqual(hypergeometric.M, 2)
        self.assertEqual(hypergeometric.n, 3)

    def test_structural_marginalization(self):

        hypergeometric = Hypergeometric(Scope([0]), 0, 0, 0)

        self.assertTrue(marginalize(hypergeometric, [1]) is not None)
        self.assertTrue(marginalize(hypergeometric, [0]) is None)


if __name__ == "__main__":
    unittest.main()
