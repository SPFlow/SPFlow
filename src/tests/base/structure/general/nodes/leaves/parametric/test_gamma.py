from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import Gamma, marginalize

import numpy as np
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        # alpha > 0
        Gamma(Scope([0]), np.nextafter(0.0, 1.0), 1.0)
        # alpha = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 0.0, 1.0)
        # alpha < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), np.nextafter(0.0, -1.0), 1.0
        )
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, Gamma, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0]), np.nan, 1.0)

        # beta > 0
        Gamma(Scope([0]), 1.0, np.nextafter(0.0, 1.0))
        # beta = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, 0.0)
        # beta < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), 1.0, np.nextafter(0.0, -1.0)
        )
        # beta = inf and beta = nan
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.inf)
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gamma, Scope([]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0, 1]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0], [1]), 1.0, 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Gamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # Gamma feature type class
        self.assertTrue(
            Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Gamma])])
        )

        # Gamma feature type instance
        self.assertTrue(
            Gamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.0, 1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Gamma.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
        )

        # conditional scope
        self.assertFalse(
            Gamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Gamma.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertEqual(gamma.alpha, 1.0)
        self.assertEqual(gamma.beta, 1.0)

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gamma])]
        )
        self.assertEqual(gamma.alpha, 1.0)
        self.assertEqual(gamma.beta, 1.0)

        gamma = Gamma.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.5, 0.5)])]
        )
        self.assertEqual(gamma.alpha, 1.5)
        self.assertEqual(gamma.beta, 0.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Gamma.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Gamma))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Gamma,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Gamma])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gamma = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)]
                )
            ]
        )
        self.assertTrue(isinstance(gamma, Gamma))
        self.assertEqual(gamma.alpha, 1.5)
        self.assertEqual(gamma.beta, 0.5)

    def test_structural_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)


if __name__ == "__main__":
    unittest.main()
