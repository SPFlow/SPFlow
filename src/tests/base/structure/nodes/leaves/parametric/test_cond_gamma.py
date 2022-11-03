from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.spn.nodes.product_node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma
from typing import Callable

import numpy as np
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        gamma = CondGamma(Scope([0], [1]))
        self.assertTrue(gamma.cond_f is None)
        gamma = CondGamma(
            Scope([0], [1]), cond_f=lambda x: {"alpha": 1.0, "beta": 1.0}
        )
        self.assertTrue(isinstance(gamma.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGamma, Scope([]))
        self.assertRaises(Exception, CondGamma, Scope([0]))
        self.assertRaises(Exception, CondGamma, Scope([0, 1], [2]))

    def test_retrieve_params(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        gamma = CondGamma(Scope([0], [1]))

        # alpha > 0
        gamma.set_cond_f(
            lambda data: {"alpha": np.nextafter(0.0, 1.0), "beta": 1.0}
        )
        alpha, beta = gamma.retrieve_params(
            np.array([[1.0]]), DispatchContext()
        )
        self.assertTrue(alpha == np.nextafter(0.0, 1.0))
        self.assertTrue(beta == 1.0)
        # alpha = 0
        gamma.set_cond_f(lambda data: {"alpha": 0.0, "beta": 1.0})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # alpha < 0
        gamma.set_cond_f(
            lambda data: {"alpha": np.nextafter(0.0, -1.0), "beta": 1.0}
        )
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # alpha = inf and alpha = nan
        gamma.set_cond_f(lambda data: {"alpha": np.inf, "beta": 1.0})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gamma.set_cond_f(lambda data: {"alpha": np.nan, "beta": 1.0})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # beta > 0
        gamma.set_cond_f(
            lambda data: {"alpha": 1.0, "beta": np.nextafter(0.0, 1.0)}
        )
        alpha, beta = gamma.retrieve_params(
            np.array([[1.0]]), DispatchContext()
        )
        self.assertTrue(alpha == 1.0)
        self.assertTrue(beta == np.nextafter(0.0, 1.0))
        # beta = 0
        gamma.set_cond_f(lambda data: {"alpha": 1.0, "beta": 0.0})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # beta < 0
        gamma.set_cond_f(
            lambda data: {"alpha": 1.0, "beta": np.nextafter(0.0, -1.0)}
        )
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # beta = inf and beta = nan
        gamma.set_cond_f(lambda data: {"alpha": 1.0, "beta": np.inf})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gamma.set_cond_f(lambda data: {"alpha": 1.0, "beta": np.nan})
        self.assertRaises(
            ValueError,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # Gamma feature type class
        self.assertTrue(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
            )
        )

        # Gamma feature type instance
        self.assertTrue(
            CondGamma.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.Gamma(1.0, 1.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondGamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondGamma.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
        )
        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
        )
        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma(1.5, 0.5)])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGamma))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGamma,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gamma = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
        )
        self.assertTrue(isinstance(gamma, CondGamma))

    def test_structural_marginalization(self):

        gamma = CondGamma(Scope([0], [2]))

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)


if __name__ == "__main__":
    unittest.main()
