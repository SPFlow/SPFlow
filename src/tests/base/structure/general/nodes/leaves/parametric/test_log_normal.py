import random
import unittest

import numpy as np

from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import LogNormal, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), stdev in (0,inf)

        # mean = +-inf and mean = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), np.nan, 1.0)

        mean = random.random()

        # stdev <= 0
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, 0.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nextafter(0.0, -1.0))
        # stdev = +-inf and stdev = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, -np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, LogNormal, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0], [1]), 0.0, 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(LogNormal.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # LogNormal feature type class
        self.assertTrue(LogNormal.accepts([FeatureContext(Scope([0]), [FeatureTypes.LogNormal])]))

        # LogNormal feature type instance
        self.assertTrue(LogNormal.accepts([FeatureContext(Scope([0]), [FeatureTypes.LogNormal(0.0, 1.0)])]))

        # invalid feature type
        self.assertFalse(LogNormal.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # conditional scope
        self.assertFalse(LogNormal.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            LogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        log_normal = LogNormal.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Continuous])])
        self.assertEqual(log_normal.mean, 0.0)
        self.assertEqual(log_normal.std, 1.0)

        log_normal = LogNormal.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.LogNormal])])
        self.assertEqual(log_normal.mean, 0.0)
        self.assertEqual(log_normal.std, 1.0)

        log_normal = LogNormal.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.LogNormal(-1.0, 1.5)])])
        self.assertEqual(log_normal.mean, -1.0)
        self.assertEqual(log_normal.std, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(LogNormal))

        # make sure leaf is correctly inferred
        self.assertEqual(
            LogNormal,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.LogNormal])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.LogNormal(mean=-1.0, std=0.5)])])
        self.assertTrue(isinstance(log_normal, LogNormal))
        self.assertEqual(log_normal.mean, -1.0)
        self.assertEqual(log_normal.std, 0.5)

    def test_structural_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)


if __name__ == "__main__":
    unittest.main()
