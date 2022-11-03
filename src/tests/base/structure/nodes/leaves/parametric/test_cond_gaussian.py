from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.product_node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from typing import Callable

import numpy as np
import unittest


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        gaussian = CondGaussian(Scope([0], [1]))
        self.assertTrue(gaussian.cond_f is None)
        gaussian = CondGaussian(
            Scope([0], [1]), cond_f=lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGaussian, Scope([]))
        self.assertRaises(Exception, CondGaussian, Scope([0]))
        self.assertRaises(Exception, CondGaussian, Scope([0, 1], [2]))

    def test_retrieve_params(self):

        # Valid parameters for Gaussian distribution: mean in (-inf,inf), stdev > 0
        gaussian = CondGaussian(Scope([0], [1]))

        # mean = inf and mean = nan
        gaussian.set_cond_f(lambda data: {"mean": np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": -np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": np.nan, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = 0 and stdev < 0
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": 0.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(
            lambda data: {"mean": 0.0, "std": np.nextafter(0.0, -1.0)}
        )
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = inf and stdev = nan
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": np.inf})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": -np.inf})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": np.nan})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # invalid scopes
        self.assertRaises(Exception, CondGaussian, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, CondGaussian, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, CondGaussian, Scope([0], [1]), 0.0, 1.0)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGaussian.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            CondGaussian.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            CondGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.Gaussian(0.0, 1.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGaussian.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondGaussian.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondGaussian.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondGaussian.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
        )
        CondGaussian.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]
        )
        CondGaussian.from_signatures(
            [
                FeatureContext(
                    Scope([0], [1]), [FeatureTypes.Gaussian(-1.0, 1.5)]
                )
            ]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGaussian.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGaussian.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGaussian.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGaussian))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGaussian,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gaussian])]
        )
        self.assertTrue(isinstance(gaussian, CondGaussian))

    def test_structural_marginalization(self):

        gaussian = CondGaussian(Scope([0], [2]))

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)


if __name__ == "__main__":
    unittest.main()
