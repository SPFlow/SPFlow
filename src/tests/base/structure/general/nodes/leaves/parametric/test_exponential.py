from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import Exponential, marginalize

import numpy as np
import unittest


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: l>0

        # l > 0
        exponential = Exponential(Scope([0]), np.nextafter(0.0, 1.0))
        # l = 0 and l < 0
        self.assertRaises(Exception, Exponential, Scope([0]), 0.0)
        self.assertRaises(
            Exception, Exponential, Scope([0]), np.nextafter(0.0, -1.0)
        )
        # l = inf and l = nan
        self.assertRaises(Exception, Exponential, Scope([0]), np.inf)
        self.assertRaises(Exception, Exponential, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Exponential, Scope([]), 1.0)
        self.assertRaises(Exception, Exponential, Scope([0, 1]), 1.0)
        self.assertRaises(Exception, Exponential, Scope([0], [1]), 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # Exponential feature type class
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
            )
        )

        # Exponential feature type instance
        self.assertTrue(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential(1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Exponential.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # conditional scope
        self.assertFalse(
            Exponential.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Exponential.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertEqual(exponential.l, 1.0)

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
        )
        self.assertEqual(exponential.l, 1.0)

        exponential = Exponential.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])]
        )
        self.assertEqual(exponential.l, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Exponential.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Exponential))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Exponential,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Exponential])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        exponential = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)])]
        )
        self.assertTrue(isinstance(exponential, Exponential))
        self.assertEqual(exponential.l, 1.5)

    def test_structural_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)


if __name__ == "__main__":
    unittest.main()