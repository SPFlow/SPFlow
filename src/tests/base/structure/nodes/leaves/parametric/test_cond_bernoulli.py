from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_bernoulli import (
    CondBernoulli,
)
from typing import Callable

import numpy as np
import unittest


class TestCondBernoulli(unittest.TestCase):
    def test_initialization(self):

        bernoulli = CondBernoulli(Scope([0], [1]))
        self.assertTrue(bernoulli.cond_f is None)
        bernoulli = CondBernoulli(Scope([0], [1]), lambda x: {"p": 0.5})
        self.assertTrue(isinstance(bernoulli.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondBernoulli, Scope([]))
        self.assertRaises(Exception, CondBernoulli, Scope([0]))
        self.assertRaises(Exception, CondBernoulli, Scope([0, 1], [2]))

    def test_retrieve_params(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        bernoulli = CondBernoulli(Scope([0], [1]))

        # p = 0
        bernoulli.set_cond_f(lambda data: {"p": 0.0})
        self.assertTrue(
            bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 0.0
        )
        # p = 1
        bernoulli.set_cond_f(lambda data: {"p": 1.0})
        self.assertTrue(
            bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 1.0
        )
        # p < 0 and p > 1
        bernoulli.set_cond_f(lambda data: {"p": np.nextafter(1.0, 2.0)})
        self.assertRaises(
            ValueError,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        bernoulli.set_cond_f(lambda data: {"p": np.nextafter(0.0, -1.0)})
        self.assertRaises(
            ValueError,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # p = inf and p = nan
        bernoulli.set_cond_f(lambda data: {"p": np.inf})
        self.assertRaises(
            ValueError,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        bernoulli.set_cond_f(lambda data: {"p": np.nan})
        self.assertRaises(
            ValueError,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type
        self.assertTrue(CondBernoulli.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # Bernoulli feature type class
        self.assertTrue(CondBernoulli.accepts([([FeatureTypes.Bernoulli], Scope([0], [1]))]))

        # Bernoulli feature type instance
        self.assertTrue(CondBernoulli.accepts([([FeatureTypes.Bernoulli(0.5)], Scope([0], [1]))]))

        # invalid feature type
        self.assertFalse(CondBernoulli.accepts([([FeatureTypes.Continuous], Scope([0], [1]))]))

        # non-conditional scope
        self.assertFalse(CondBernoulli.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # scope length does not match number of types
        self.assertFalse(CondBernoulli.accepts([([FeatureTypes.Discrete], Scope([0, 1], [2]))]))

        # multivariate signature
        self.assertFalse(CondBernoulli.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1], [2]))]))

    def test_initialization_from_signatures(self):

        CondBernoulli.from_signatures([([FeatureTypes.Discrete], Scope([0], [1]))])
        CondBernoulli.from_signatures([([FeatureTypes.Bernoulli], Scope([0], [1]))])
        CondBernoulli.from_signatures([([FeatureTypes.Bernoulli(p=0.75)], Scope([0], [1]))])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, CondBernoulli.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # non-conditional scope
        self.assertRaises(ValueError, CondBernoulli.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, CondBernoulli.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1], [2]))])

        # multivariate signature
        self.assertRaises(ValueError, CondBernoulli.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1], [2]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondBernoulli))

        # make sure leaf is correctly inferred
        self.assertEqual(CondBernoulli, AutoLeaf.infer([([FeatureTypes.Bernoulli], Scope([0], [1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf([([FeatureTypes.Bernoulli], Scope([0], [1]))])
        self.assertTrue(isinstance(bernoulli, CondBernoulli))

    def test_conditional_marginalization(self):

        bernoulli = CondBernoulli(Scope([0], [2]))

        self.assertTrue(marginalize(bernoulli, [1]) is not None)
        self.assertTrue(marginalize(bernoulli, [0]) is None)


if __name__ == "__main__":
    unittest.main()
