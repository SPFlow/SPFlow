from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
)
from typing import Callable

import numpy as np
import unittest


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        poisson = CondPoisson(Scope([0], [1]))
        self.assertTrue(poisson.cond_f is None)
        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 0.5})
        self.assertTrue(isinstance(poisson.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondPoisson, Scope([]))
        self.assertRaises(Exception, CondPoisson, Scope([0]))
        self.assertRaises(Exception, CondPoisson, Scope([0, 1], [2]))

    def test_retrieve_params(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in scipy)

        poisson = CondPoisson(Scope([0], [1]))

        # l = 0
        poisson.set_cond_f(lambda data: {"l": 0.0})
        self.assertTrue(
            poisson.retrieve_params(np.array([[1.0]]), DispatchContext()) == 0.0
        )
        # l > 0
        poisson.set_cond_f(lambda data: {"l": np.nextafter(0.0, 1.0)})
        self.assertTrue(
            poisson.retrieve_params(np.array([[1.0]]), DispatchContext())
            == np.nextafter(0.0, 1.0)
        )
        # l = -inf and l = inf
        poisson.set_cond_f(lambda data: {"l": -np.inf})
        self.assertRaises(
            ValueError,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        poisson.set_cond_f(lambda data: {"l": np.inf})
        self.assertRaises(
            ValueError,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # l = nan
        poisson.set_cond_f(lambda data: {"l": np.nan})
        self.assertRaises(
            ValueError,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(CondPoisson.accepts([([FeatureTypes.Discrete], Scope([0], [1]))]))

        # Poisson feature type class
        self.assertTrue(CondPoisson.accepts([([FeatureTypes.Poisson], Scope([0], [1]))]))

        # Poisson feature type instance
        self.assertTrue(CondPoisson.accepts([([FeatureTypes.Poisson(1.0)], Scope([0], [1]))]))

        # invalid feature type
        self.assertFalse(CondPoisson.accepts([([FeatureTypes.Continuous], Scope([0], [1]))]))

        # non-conditional scope
        self.assertFalse(CondPoisson.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # scope length does not match number of types
        self.assertFalse(CondPoisson.accepts([([FeatureTypes.Discrete], Scope([0, 1], [2]))]))

        # multivariate signature
        self.assertFalse(CondPoisson.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        poisson = CondPoisson.from_signatures([([FeatureTypes.Discrete], Scope([0], [1]))])
        poisson = CondPoisson.from_signatures([([FeatureTypes.Poisson], Scope([0], [1]))])
        poisson = CondPoisson.from_signatures([([FeatureTypes.Poisson(l=1.5)], Scope([0], [1]))])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, CondPoisson.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # non-conditional scope
        self.assertRaises(ValueError, CondPoisson.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, CondPoisson.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1], [2]))])

        # multivariate signature
        self.assertRaises(ValueError, CondPoisson.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondPoisson))

        # make sure leaf is correctly inferred
        self.assertEqual(CondPoisson, AutoLeaf.infer([([FeatureTypes.Poisson], Scope([0], [1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf([([FeatureTypes.Poisson(l=1.5)], Scope([0], [1]))])
        self.assertTrue(isinstance(poisson, CondPoisson))

    def test_structural_marginalization(self):

        poisson = CondPoisson(Scope([0], [2]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)


if __name__ == "__main__":
    unittest.main()
