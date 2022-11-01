from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial,
)

import numpy as np
import unittest


class TestNegativeBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Negative Binomial distribution: p in (0,1], n in N U {0}

        # p = 1
        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)
        # p = 0
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, 0.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception, NegativeBinomial, Scope([0]), 1, np.nextafter(1.0, 2.0)
        )
        self.assertRaises(
            Exception, NegativeBinomial, Scope([0]), 1, np.nextafter(0.0, -1.0)
        )
        # p = +-inf and p = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, -np.inf)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), 1, np.nan)

        # n = 0
        NegativeBinomial(Scope([0]), 0.0, 1.0)
        # n < 0
        self.assertRaises(
            Exception,
            NegativeBinomial,
            Scope([0]),
            np.nextafter(0.0, -1.0),
            1.0,
        )
        # n = inf and = nan
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, NegativeBinomial, Scope([0]), np.nan, 1.0)

        # invalid scopes
        self.assertRaises(Exception, NegativeBinomial, Scope([]), 1, 0.5)
        self.assertRaises(Exception, NegativeBinomial, Scope([0, 1]), 1, 0.5)
        self.assertRaises(Exception, NegativeBinomial, Scope([0], [1]), 1, 0.5)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            NegativeBinomial.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            NegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            NegativeBinomial.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            NegativeBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            )
        )

        # multivariate signature
        self.assertFalse(
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

    def test_initialization_from_signatures(self):

        negative_binomial = NegativeBinomial.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])]
        )
        self.assertEqual(negative_binomial.n, 3)
        self.assertEqual(negative_binomial.p, 0.5)

        negative_binomial = NegativeBinomial.from_signatures(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)]
                )
            ]
        )
        self.assertEqual(negative_binomial.n, 3)
        self.assertEqual(negative_binomial.p, 0.75)

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            NegativeBinomial.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(NegativeBinomial))

        # make sure leaf is correctly inferred
        self.assertEqual(
            NegativeBinomial,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.NegativeBinomial(n=3)]
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        negative_binomial = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.NegativeBinomial(n=3, p=0.75)]
                )
            ]
        )
        self.assertTrue(isinstance(negative_binomial, NegativeBinomial))
        self.assertEqual(negative_binomial.n, 3)
        self.assertEqual(negative_binomial.p, 0.75)

    def test_structural_marginalization(self):

        negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

        self.assertTrue(marginalize(negative_binomial, [1]) is not None)
        self.assertTrue(marginalize(negative_binomial, [0]) is None)


if __name__ == "__main__":
    unittest.main()
