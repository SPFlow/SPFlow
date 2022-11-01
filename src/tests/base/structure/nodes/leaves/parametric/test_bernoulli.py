from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli

import numpy as np
import unittest


class TestBernoulli(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        # p = 0
        bernoulli = Bernoulli(Scope([0]), 0.0)
        # p = 1
        bernoulli = Bernoulli(Scope([0]), 1.0)
        # p < 0 and p > 1
        self.assertRaises(
            Exception, Bernoulli, Scope([0]), np.nextafter(1.0, 2.0)
        )
        self.assertRaises(
            Exception, Bernoulli, Scope([0]), np.nextafter(0.0, -1.0)
        )
        # p = inf and p = nan
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.inf)
        self.assertRaises(Exception, Bernoulli, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Bernoulli, Scope([]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Bernoulli, Scope([0], [1]), 0.5)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Bernoulli feature type class
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.5)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Bernoulli.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            Bernoulli.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Bernoulli.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        )
        self.assertEqual(bernoulli.p, 0.5)

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
        )
        self.assertEqual(bernoulli.p, 0.5)

        bernoulli = Bernoulli.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])]
        )
        self.assertEqual(bernoulli.p, 0.75)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Bernoulli.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Bernoulli))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Bernoulli,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])]
        )
        self.assertTrue(isinstance(bernoulli, Bernoulli))
        self.assertEqual(bernoulli.p, 0.75)

    def test_structural_marginalization(self):

        bernoulli = Bernoulli(Scope([0]), 0.5)

        self.assertTrue(marginalize(bernoulli, [1]) is not None)
        self.assertTrue(marginalize(bernoulli, [0]) is None)


if __name__ == "__main__":
    unittest.main()
