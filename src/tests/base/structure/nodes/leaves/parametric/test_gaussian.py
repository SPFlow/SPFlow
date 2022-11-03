from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.product_node import marginalize
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian

import numpy as np
import unittest
import random
import math


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gaussian distribution: mean in (-inf,inf), stdev > 0

        mean = random.random()

        # mean = inf and mean = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), np.nan, 1.0)

        # stdev = 0 and stdev < 0
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, 0.0)
        self.assertRaises(
            Exception, Gaussian, Scope([0]), mean, np.nextafter(0.0, -1.0)
        )
        # stdev = inf and stdev = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gaussian, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0], [1]), 0.0, 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            Gaussian.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            Gaussian.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            Gaussian.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Gaussian.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # conditional scope
        self.assertFalse(
            Gaussian.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Gaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gaussian = Gaussian.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertEqual(gaussian.mean, 0.0)
        self.assertEqual(gaussian.std, 1.0)

        gaussian = Gaussian.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]
        )
        self.assertEqual(gaussian.mean, 0.0)
        self.assertEqual(gaussian.std, 1.0)

        gaussian = Gaussian.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Gaussian(-1.0, 1.5)])]
        )
        self.assertEqual(gaussian.mean, -1.0)
        self.assertEqual(gaussian.std, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Gaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Gaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Gaussian,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Gaussian])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Gaussian(mean=-1.0, std=0.5)]
                )
            ]
        )
        self.assertTrue(isinstance(gaussian, Gaussian))
        self.assertEqual(gaussian.mean, -1.0)
        self.assertEqual(gaussian.std, 0.5)

    def test_structural_marginalization(self):

        gaussian = Gaussian(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)


if __name__ == "__main__":
    unittest.main()
