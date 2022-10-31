from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform

import numpy as np
import unittest
import random


class TestUniform(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Uniform distribution: a<b

        # start = end
        start_end = random.random()
        self.assertRaises(Exception, Uniform, Scope([0]), start_end, start_end)
        # start > end
        self.assertRaises(
            Exception,
            Uniform,
            Scope([0]),
            start_end,
            np.nextafter(start_end, -1.0),
        )
        # start = +-inf and start = nan
        self.assertRaises(Exception, Uniform, Scope([0]), np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), -np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), np.nan, 0.0)

        # end = +-inf and end = nan
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, -np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Uniform, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0], [1]))

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(Uniform.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # Bernoulli feature type class (should reject)
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform], Scope([0]))]))

        # Bernoulli feature type instance
        self.assertTrue(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Uniform.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # conditional scope
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0), FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        uniform = Uniform.from_signatures([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))])
        self.assertEqual(uniform.start, -1.0)
        self.assertEqual(uniform.end, 2.0)

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # Bernoulli feature type class
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Uniform], Scope([0]))])

        # invalid feature type
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Uniform))

        # make sure leaf is correctly inferred
        self.assertEqual(Uniform, AutoLeaf.infer([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        uniform = AutoLeaf([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))])
        self.assertTrue(isinstance(uniform, Uniform))
        self.assertEqual(uniform.start, -1.0)
        self.assertEqual(uniform.end, 2.0)

    def test_structural_marginalization(self):

        uniform = Uniform(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(uniform, [1]) is not None)
        self.assertTrue(marginalize(uniform, [0]) is None)


if __name__ == "__main__":
    unittest.main()
