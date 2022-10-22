from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_geometric import CondGeometric
from typing import Callable

import numpy as np
import unittest


class TestGeometric(unittest.TestCase):
    def test_initialization(self):

        geometric = CondGeometric(Scope([0]))
        self.assertTrue(geometric.cond_f is None)
        geometric = CondGeometric(Scope([0]), lambda x: {'l': 0.5})
        self.assertTrue(isinstance(geometric.cond_f, Callable))
        
        # invalid scopes
        self.assertRaises(Exception, CondGeometric, Scope([]))
        self.assertRaises(Exception, CondGeometric, Scope([0, 1]))
        self.assertRaises(Exception, CondGeometric, Scope([0],[1]))

    def test_retrieve_params(self):

        # Valid parameters for Geometric distribution: p in (0,1]

        geometric = CondGeometric(Scope([0]))

        # p = 0
        geometric.set_cond_f(lambda data: {'p': 0.0})
        self.assertRaises(ValueError, geometric.retrieve_params, np.array([[1.0]]), DispatchContext())
        # p = inf and p = nan
        geometric.set_cond_f(lambda data: {'p': np.inf})
        self.assertRaises(ValueError, geometric.retrieve_params, np.array([[1.0]]), DispatchContext())
        geometric.set_cond_f(lambda data: {'p': -np.inf})
        self.assertRaises(ValueError, geometric.retrieve_params, np.array([[1.0]]), DispatchContext())
        geometric.set_cond_f(lambda data: {'p': np.nan})
        self.assertRaises(ValueError, geometric.retrieve_params, np.array([[1.0]]), DispatchContext())

    def test_structural_marginalization(self):
        
        geometric = CondGeometric(Scope([0]), 0.5)

        self.assertTrue(marginalize(geometric, [1]) is not None)
        self.assertTrue(marginalize(geometric, [0]) is None)

if __name__ == "__main__":
    unittest.main()
