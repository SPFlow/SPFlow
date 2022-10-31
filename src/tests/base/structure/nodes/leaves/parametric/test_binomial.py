from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial

import numpy as np
import unittest


class TestBinomial(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Binomial distribution: p in [0,1], n in N U {0}

        # p = 0
        binomial = Binomial(Scope([0]), 1, 0.0)
        # p = 1
        binomial = Binomial(Scope([0]), 1, 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception, Binomial, Scope([0]), 1, np.nextafter(1.0, 2.0)
        )
        self.assertRaises(
            Exception, Binomial, Scope([0]), 1, np.nextafter(0.0, -1.0)
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.inf)
        self.assertRaises(Exception, Binomial, Scope([0]), 1, np.nan)

        # n = 0
        binomial = Binomial(Scope([0]), 0, 0.5)
        # n < 0
        self.assertRaises(Exception, Binomial, Scope([0]), -1, 0.5)
        # n float
        self.assertRaises(Exception, Binomial, Scope([0]), 0.5, 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, Binomial, Scope([0]), np.inf, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0]), np.nan, 0.5)

        # invalid scopes
        self.assertRaises(Exception, Binomial, Scope([]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0, 1]), 1, 0.5)
        self.assertRaises(Exception, Binomial, Scope([0], [1]), 1, 0.5)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(Binomial.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # Bernoulli feature type class (should reject)
        self.assertFalse(Binomial.accepts([([FeatureTypes.Binomial], Scope([0]))]))

        # Bernoulli feature type instance
        self.assertTrue(Binomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Binomial.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # conditional scope
        self.assertFalse(Binomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Binomial.accepts([([FeatureTypes.Binomial(n=3)], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Binomial.accepts([([FeatureTypes.Binomial(n=3), FeatureTypes.Binomial(n=3)], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        binomial = Binomial.from_signatures([([FeatureTypes.Binomial(n=3)], Scope([0]))])
        self.assertEqual(binomial.n, 3)
        self.assertEqual(binomial.p, 0.5)

        binomial = Binomial.from_signatures([([FeatureTypes.Binomial(n=3, p=0.75)], Scope([0]))])
        self.assertEqual(binomial.n, 3)
        self.assertEqual(binomial.p, 0.75)

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # Bernoulli feature type class
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Binomial], Scope([0]))])

        # invalid feature type
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Binomial.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Binomial))

        # make sure leaf is correctly inferred
        self.assertEqual(Binomial, AutoLeaf.infer([([FeatureTypes.Binomial(n=3)], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        binomial = AutoLeaf([([FeatureTypes.Binomial(n=3, p=0.75)], Scope([0]))])
        self.assertTrue(isinstance(binomial, Binomial))
        self.assertEqual(binomial.n, 3)
        self.assertEqual(binomial.p, 0.75)

    def test_structural_marginalization(self):

        binomial = Binomial(Scope([0]), 1, 0.5)

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)


if __name__ == "__main__":
    unittest.main()
