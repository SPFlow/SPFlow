from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.node import marginalize
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
        self.assertFalse(
            Uniform.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # feature type instance
        self.assertTrue(
            Uniform.accepts(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            Uniform.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # conditional scope
        self.assertFalse(
            Uniform.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]),
                        [FeatureTypes.Uniform(start=-1.0, end=2.0)],
                    )
                ]
            )
        )

        # multivariate signature
        self.assertFalse(
            Uniform.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [
                            FeatureTypes.Uniform(start=-1.0, end=2.0),
                            FeatureTypes.Uniform(start=-1.0, end=2.0),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        uniform = Uniform.from_signatures(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]
                )
            ]
        )
        self.assertEqual(uniform.start, -1.0)
        self.assertEqual(uniform.end, 2.0)

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            Uniform.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            Uniform.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            Uniform.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Uniform.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Uniform))

        # make sure leaf is correctly inferred
        self.assertEqual(
            Uniform,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]
                    )
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        uniform = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]
                )
            ]
        )
        self.assertTrue(isinstance(uniform, Uniform))
        self.assertEqual(uniform.start, -1.0)
        self.assertEqual(uniform.end, 2.0)

    def test_structural_marginalization(self):

        uniform = Uniform(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(uniform, [1]) is not None)
        self.assertTrue(marginalize(uniform, [0]) is None)


if __name__ == "__main__":
    unittest.main()
