import unittest

import numpy as np
import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import marginalize
from spflow.tensorly.structure.general.nodes.leaves import Poisson
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in scipy)

        # l = 0
        Poisson(Scope([0]), 0.0)
        # l > 0
        Poisson(Scope([0]), np.nextafter(0.0, 1.0))
        # l = -inf and l = inf
        self.assertRaises(Exception, Poisson, Scope([0]), -tl.inf)
        self.assertRaises(Exception, Poisson, Scope([0]), tl.inf)
        # l = nan
        self.assertRaises(Exception, Poisson, Scope([0]), tl.nan)

        # invalid scopes
        self.assertRaises(Exception, Poisson, Scope([]), 1)
        self.assertRaises(Exception, Poisson, Scope([0, 1]), 1)
        self.assertRaises(Exception, Poisson, Scope([0], [1]), 1)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # Poisson feature type class
        self.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Poisson])]))

        # Poisson feature type instance
        self.assertTrue(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Poisson(1.0)])]))

        # invalid feature type
        self.assertFalse(Poisson.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # conditional scope
        self.assertFalse(Poisson.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            Poisson.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Discrete])])
        self.assertEqual(poisson.l, 1.0)

        poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Poisson])])
        self.assertEqual(poisson.l, 1.0)

        poisson = Poisson.from_signatures([FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])])
        self.assertEqual(poisson.l, 1.5)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Poisson.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Poisson))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Poisson,
            AutoLeaf.infer([FeatureContext(Scope([0]), [FeatureTypes.Poisson])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf([FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)])])
        self.assertTrue(isinstance(poisson, Poisson))
        self.assertEqual(poisson.l, 1.5)

    def test_structural_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)


if __name__ == "__main__":
    unittest.main()
