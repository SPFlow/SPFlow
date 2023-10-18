import unittest

import numpy as np

from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn import Categorical, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope

class TestCategorical(unittest.TestCase):
    def test_initialization(self):
        # valid parameters for Categorical distribution: p \in R^n, \all p_i \in p: p_i \in [0,1], \sum_i p_i = 1

        categorical = Categorical(Scope([0]), k=1, p=[1.0])
        categorical = Categorical(Scope([0]), k=4, p=[0.1, 0.2, 0.3, 0.4])

        # p_i < 0, p_i > 1
        self.assertRaises(Exception, Categorical, Scope([0]), 1, [np.nextafter(1.0, 2.0)])
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [1.0, np.nextafter(0.0, -1.0)])
        # \sum p_i != 1
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [0.5, 0.4])
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [0.5, 0.6])
        # p not a list/array
        self.assertRaises(Exception, Categorical, Scope([0]), 2, 1.0)
        # k and |p| not matching
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [1.0])
        # p = inf, p = nan
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [1.0, np.inf])
        self.assertRaises(Exception, Categorical, Scope([0]), 2, [1.0, np.nan])

        # invalid scopes
        self.assertRaises(Exception, Categorical, Scope([]), 2, [0.5, 0.5])
        self.assertRaises(Exception, Categorical, Scope([0, 1]), 2, [0.5, 0.5])
        self.assertRaises(Exception, Categorical, Scope([0], [1]), [0.5, 0.5])


    def test_accept(self):

        # discrete meta type
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))
        
        # Categorical feature type class
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical])]))

        # Categorical feature type instance
        self.assertTrue(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])])
                        )
        # invalid feature type
        self.assertFalse(Categorical.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # conditional scope
        self.assertFalse(Categorical.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(Categorical.accepts([FeatureContext(Scope([0, 1]), [FeatureTypes.Discrete, FeatureTypes.Discrete])]))

    
    def test_initialization_from_signature(self):

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
        self.assertEqual(categorical.k, 2)
        self.assertListEqual(categorical.p.tolist(), [0.5, 0.5])

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Categorical])])
        self.assertEqual(categorical.k, 2)
        self.assertListEqual(categorical.p.tolist(), [0.5, 0.5])

        categorical = Categorical.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=4, p=[0.1, 0.2, 0.3, 0.4])])])
        self.assertEqual(categorical.k, 4)
        self.assertListEqual(categorical.p.tolist(), [0.1, 0.2, 0.3, 0.4])

        # invalid arguments
        
        #invalid feature type
        self.assertRaises(ValueError, Categorical.from_signatures, [FeatureContext(Scope([0]), [FeatureTypes.Continuous])])

        # conditional scope
        self.assertRaises(ValueError, Categorical.from_signatures, [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])])

        # multivariate signature
        self.assertRaises(ValueError, Categorical.from_signatures, [FeatureContext(Scope([0, 1]), [FeatureTypes.Discrete, FeatureTypes.Discrete])])


    def test_autoleaf(self):
        
        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Categorical))

        # make sure is correctly inferred
        self.assertEqual(Categorical, AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Categorical])]))

        # make sure AutoLeaf can return correctly instantiated object
        categorical = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])])
        self.assertTrue(isinstance(categorical, Categorical))
        self.assertEqual(categorical.k, 2)
        self.assertListEqual(categorical.p.tolist(), [0.5, 0.5])


    def test_structural_marginalization(self):

        categorical = Categorical(Scope([0]), 2, [0.5, 0.5])
        self.assertTrue(marginalize(categorical, [1]) is not None)
        self.assertTrue(marginalize(categorical, [0]) is None)


if __name__ == "__main__":
    unittest.main()
