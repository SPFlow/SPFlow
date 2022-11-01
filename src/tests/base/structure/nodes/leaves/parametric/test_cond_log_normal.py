from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from typing import Callable

import numpy as np
import unittest


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        log_normal = CondLogNormal(Scope([0], [1]))
        self.assertTrue(log_normal.cond_f is None)
        log_normal = CondLogNormal(
            Scope([0], [1]), cond_f=lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(log_normal.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondLogNormal, Scope([]))
        self.assertRaises(Exception, CondLogNormal, Scope([0]))
        self.assertRaises(Exception, CondLogNormal, Scope([0, 1], [2]))

    def test_retrieve_params(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), stdev in (0,inf)

        log_normal = CondLogNormal(Scope([0], [1]))

        # mean = inf and mean = nan
        log_normal.set_cond_f(lambda data: {"mean": np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": -np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": np.nan, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = 0 and stdev < 0
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": 0.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {"mean": 0.0, "std": np.nextafter(0.0, -1.0)}
        )
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = inf and stdev = nan
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": np.inf})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": -np.inf})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": np.nan})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # LogNormal feature type class
        self.assertTrue(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
            )
        )

        # LogNormal feature type instance
        self.assertTrue(
            CondLogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.LogNormal(0.0, 1.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondLogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondLogNormal.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
        )
        CondLogNormal.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
        )
        CondLogNormal.from_signatures(
            [
                FeatureContext(
                    Scope([0], [1]), [FeatureTypes.LogNormal(-1.0, 1.5)]
                )
            ]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondLogNormal))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondLogNormal,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
        )
        self.assertTrue(isinstance(log_normal, CondLogNormal))

    def test_structural_marginalization(self):

        log_normal = CondLogNormal(Scope([0], [2]))

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)


if __name__ == "__main__":
    unittest.main()
