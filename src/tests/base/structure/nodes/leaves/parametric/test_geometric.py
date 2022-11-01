from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric

import numpy as np
import unittest


class TestGeometric(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Geometric distribution: p in (0,1]

        # p = 0
        self.assertRaises(Exception, Geometric, Scope([0]), 0.0)
        # p = inf and p = nan
        self.assertRaises(Exception, Geometric, Scope([0]), np.inf)
        self.assertRaises(Exception, Geometric, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Geometric, Scope([]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0], [1]), 0.5)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # Geometric feature type class
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
            )
        )

        # Geometric feature type instance
        self.assertTrue(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            Geometric.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # conditional scope
        self.assertFalse(
            Geometric.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            Geometric.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        )
        self.assertEqual(geometric.p, 0.5)

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
        )
        self.assertEqual(geometric.p, 0.5)

        geometric = Geometric.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])]
        )
        self.assertEqual(geometric.p, 0.75)

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Geometric.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Geometric))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Geometric,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.Geometric])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf(
            [FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)])]
        )
        self.assertTrue(isinstance(geometric, Geometric))
        self.assertEqual(geometric.p, 0.75)

    def test_structural_marginalization(self):

        geometric = Geometric(Scope([0]), 0.5)

        self.assertTrue(marginalize(geometric, [1]) is not None)
        self.assertTrue(marginalize(geometric, [0]) is None)


if __name__ == "__main__":
    unittest.main()
