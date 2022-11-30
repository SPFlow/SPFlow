import unittest
from typing import Callable

import numpy as np

from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.general.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from spflow.base.structure.spn.nodes.product_node import marginalize
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext


class TestCondBinomial(unittest.TestCase):
    def test_initialization(self):

        binomial = CondBinomial(Scope([0], [1]), n=2)
        self.assertTrue(binomial.cond_f is None)
        binomial = CondBinomial(Scope([0], [1]), n=2, cond_f=lambda x: {"p": 0.5})
        self.assertTrue(isinstance(binomial.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondBinomial, Scope([]), n=2)
        self.assertRaises(Exception, CondBinomial, Scope([0]), n=2)
        self.assertRaises(Exception, CondBinomial, Scope([0, 1], [2]), n=2)

        # Valid parameters for Binomial distribution: n in N U {0}

        # n = 0
        binomial = CondBinomial(Scope([0], [1]), 0)
        # n < 0
        self.assertRaises(Exception, CondBinomial, Scope([0], [1]), -1)
        # n float
        self.assertRaises(Exception, CondBinomial, Scope([0], [1]), 0.5)
        # n = inf and n = nan
        self.assertRaises(Exception, CondBinomial, Scope([0], [1]), np.inf)
        self.assertRaises(Exception, CondBinomial, Scope([0], [1]), np.nan)

    def test_retrieve_params(self):

        # Valid parameters for Binomial distribution: p in [0,1]

        binomial = CondBinomial(Scope([0], [1]), n=2)

        # p = 0
        binomial.set_cond_f(lambda data: {"p": 0.0})
        self.assertTrue(binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 0.0)
        # p = 1
        binomial.set_cond_f(lambda data: {"p": 1.0})
        self.assertTrue(binomial.retrieve_params(np.array([[1.0]]), DispatchContext()) == 1.0)
        # p < 0 and p > 1
        binomial.set_cond_f(lambda data: {"p": np.nextafter(1.0, 2.0)})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        binomial.set_cond_f(lambda data: {"p": np.nextafter(0.0, -1.0)})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # p = inf and p = nan
        binomial.set_cond_f(lambda data: {"p": np.inf})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        binomial.set_cond_f(lambda data: {"p": np.nan})
        self.assertRaises(
            ValueError,
            binomial.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # feature type instance
        self.assertTrue(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]))

        # invalid feature type
        self.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # non-conditional scope
        self.assertFalse(CondBinomial.accepts([FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])]))

        # multivariate signature
        self.assertFalse(
            CondBinomial.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [
                            FeatureTypes.Binomial(n=3),
                            FeatureTypes.Binomial(n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])])
        CondBinomial.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3, p=0.75)])])

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            CondBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondBinomial.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            CondBinomial.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondBinomial.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondBinomial))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondBinomial,
            AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        binomial = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])])
        self.assertTrue(isinstance(binomial, CondBinomial))
        self.assertEqual(binomial.n, 3)

    def test_structural_marginalization(self):

        binomial = CondBinomial(Scope([0], [2]), 1, lambda data: {"p": 0.5})

        self.assertTrue(marginalize(binomial, [1]) is not None)
        self.assertTrue(marginalize(binomial, [0]) is None)


if __name__ == "__main__":
    unittest.main()
