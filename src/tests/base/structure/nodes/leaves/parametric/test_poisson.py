from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson

import numpy as np
import unittest


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in scipy)

        # l = 0
        Poisson(Scope([0]), 0.0)
        # l > 0
        Poisson(Scope([0]), np.nextafter(0.0, 1.0))
        # l = -inf and l = inf
        self.assertRaises(Exception, Poisson, Scope([0]), -np.inf)
        self.assertRaises(Exception, Poisson, Scope([0]), np.inf)
        # l = nan
        self.assertRaises(Exception, Poisson, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Poisson, Scope([]), 1)
        self.assertRaises(Exception, Poisson, Scope([0, 1]), 1)
        self.assertRaises(Exception, Poisson, Scope([0], [1]), 1)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(Poisson.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # Poisson feature type class
        self.assertTrue(Poisson.accepts([([FeatureTypes.Poisson], Scope([0]))]))

        # Poisson feature type instance
        self.assertTrue(Poisson.accepts([([FeatureTypes.Poisson(1.0)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Poisson.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # conditional scope
        self.assertFalse(Poisson.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Poisson.accepts([([FeatureTypes.Discrete], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Poisson.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        poisson = Poisson.from_signatures([([FeatureTypes.Discrete], Scope([0]))])
        self.assertEqual(poisson.l, 1.0)

        poisson = Poisson.from_signatures([([FeatureTypes.Poisson], Scope([0]))])
        self.assertEqual(poisson.l, 1.0)
    
        poisson = Poisson.from_signatures([([FeatureTypes.Poisson(l=1.5)], Scope([0]))])
        self.assertEqual(poisson.l, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, Poisson.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Poisson.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Poisson.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Poisson.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Poisson))

        # make sure leaf is correctly inferred
        self.assertEqual(Poisson, AutoLeaf.infer([([FeatureTypes.Poisson], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf([([FeatureTypes.Poisson(l=1.5)], Scope([0]))])
        self.assertTrue(isinstance(poisson, Poisson))
        self.assertEqual(poisson.l, 1.5)

    def test_structural_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)


if __name__ == "__main__":
    unittest.main()
