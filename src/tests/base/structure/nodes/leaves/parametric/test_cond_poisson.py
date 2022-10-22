from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import CondPoisson
from typing import Callable

import numpy as np
import unittest
import random


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        poisson = CondPoisson(Scope([0]))
        self.assertTrue(poisson.cond_f is None)
        poisson = CondPoisson(Scope([0]), cond_f=lambda x: {'l': 0.5})
        self.assertTrue(isinstance(poisson.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondPoisson, Scope([]))
        self.assertRaises(Exception, CondPoisson, Scope([0, 1]))
        self.assertRaises(Exception, CondPoisson, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in scipy)

        poisson = CondPoisson(Scope([0]))

        # l = 0
        poisson.set_cond_f(lambda data: {'l': 0.0})
        self.assertTrue(poisson.retrieve_params(np.array([[1.0]]), DispatchContext()) == 0.0)
        # l > 0
        poisson.set_cond_f(lambda data: {'l': np.nextafter(0.0, 1.0)})
        self.assertTrue(poisson.retrieve_params(np.array([[1.0]]), DispatchContext()) == np.nextafter(0.0, 1.0))
        # l = -inf and l = inf
        poisson.set_cond_f(lambda data: {'l': -np.inf})
        self.assertRaises(ValueError, poisson.retrieve_params, np.array([[1.0]]), DispatchContext())
        poisson.set_cond_f(lambda data: {'l': np.inf})
        self.assertRaises(ValueError, poisson.retrieve_params, np.array([[1.0]]), DispatchContext())
        # l = nan
        poisson.set_cond_f(lambda data: {'l': np.nan})
        self.assertRaises(ValueError, poisson.retrieve_params, np.array([[1.0]]), DispatchContext())

        # invalid scopes
        self.assertRaises(Exception, CondPoisson, Scope([]), 1)
        self.assertRaises(Exception, CondPoisson, Scope([0, 1]), 1)
        self.assertRaises(Exception, CondPoisson, Scope([0],[1]), 1)

    def test_structural_marginalization(self):
        
        poisson = CondPoisson(Scope([0]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)

if __name__ == "__main__":
    unittest.main()
