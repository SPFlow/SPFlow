from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
)
from typing import Callable

import numpy as np
import unittest


class TestCondExponential(unittest.TestCase):
    def test_initialization(self):

        binomial = CondExponential(Scope([0]))
        self.assertTrue(binomial.cond_f is None)
        binomial = CondExponential(Scope([0]), cond_f=lambda x: {"l": 0.5})
        self.assertTrue(isinstance(binomial.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondExponential, Scope([]))
        self.assertRaises(Exception, CondExponential, Scope([0, 1]))
        self.assertRaises(Exception, CondExponential, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Exponential distribution: l>0

        exponential = CondExponential(Scope([0]))

        # l > 0
        exponential.set_cond_f(lambda data: {"l": np.nextafter(0.0, 1.0)})
        self.assertTrue(
            exponential.retrieve_params(np.array([[1.0]]), DispatchContext())
            == np.nextafter(0.0, 1.0)
        )
        # l = 0 and l < 0
        exponential.set_cond_f(lambda data: {"l": 0.0})
        self.assertRaises(
            ValueError,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        exponential.set_cond_f(lambda data: {"l": np.nextafter(0.0, -1.0)})
        self.assertRaises(
            ValueError,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # l = inf and l = nan
        exponential.set_cond_f(lambda data: {"l": np.inf})
        self.assertRaises(
            ValueError,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        exponential.set_cond_f(lambda data: {"l": np.nan})
        self.assertRaises(
            ValueError,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # invalid scopes
        self.assertRaises(Exception, CondExponential, Scope([]), 1.0)
        self.assertRaises(Exception, CondExponential, Scope([0, 1]), 1.0)
        self.assertRaises(Exception, CondExponential, Scope([0], [1]), 1.0)

    def test_structural_marginalization(self):

        exponential = CondExponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)


if __name__ == "__main__":
    unittest.main()
