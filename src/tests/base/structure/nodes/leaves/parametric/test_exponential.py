from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential,
)

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
        self.assertTrue(Exponential.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # Exponential feature type class
        self.assertTrue(Exponential.accepts([([FeatureTypes.Exponential], Scope([0]))]))

        # Exponential feature type instance
        self.assertTrue(Exponential.accepts([([FeatureTypes.Exponential(1.0)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Exponential.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # conditional scope
        self.assertFalse(Exponential.accepts([([FeatureTypes.Continuous], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Exponential.accepts([([FeatureTypes.Continuous], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Exponential.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        exponential = Exponential.from_signatures([([FeatureTypes.Continuous], Scope([0]))])
        self.assertEqual(exponential.l, 1.0)

        exponential = Exponential.from_signatures([([FeatureTypes.Exponential], Scope([0]))])
        self.assertEqual(exponential.l, 1.0)
    
        exponential = Exponential.from_signatures([([FeatureTypes.Exponential(l=1.5)], Scope([0]))])
        self.assertEqual(exponential.l, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, Exponential.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Exponential.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Exponential.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Exponential.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Exponential))

        # make sure leaf is correctly inferred
        self.assertEqual(Exponential, AutoLeaf.infer([([FeatureTypes.Exponential], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        exponential = AutoLeaf([([FeatureTypes.Exponential(l=1.5)], Scope([0]))])
        self.assertTrue(isinstance(exponential, Exponential))
        self.assertEqual(exponential.l, 1.5)

    def test_structural_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)


if __name__ == "__main__":
    unittest.main()
