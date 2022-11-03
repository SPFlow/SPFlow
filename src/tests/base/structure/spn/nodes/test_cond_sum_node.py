from spflow.base.structure.spn.nodes.cond_sum_node import (
    SPNCondSumNode,
    marginalize,
)
from spflow.base.structure.spn.nodes.product_node import (
    SPNProductNode,
    marginalize,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from .dummy_node import DummyNode
from typing import Callable

import numpy as np
import unittest


class TestSumNode(unittest.TestCase):
    def test_initialization(self):

        sum_node = SPNCondSumNode(
            [DummyNode(Scope([0])), DummyNode(Scope([0]))]
        )
        self.assertTrue(sum_node.cond_f is None)
        sum_node = SPNCondSumNode(
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            lambda x: {"weights": [0.5, 0.5]},
        )
        self.assertTrue(isinstance(sum_node.cond_f, Callable))

        # empty children
        self.assertRaises(ValueError, SPNCondSumNode, [], [])
        # non-Module children
        self.assertRaises(
            ValueError, SPNCondSumNode, [DummyNode(Scope([0])), 0]
        )
        # children with different scopes
        self.assertRaises(
            ValueError,
            SPNCondSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([1]))],
        )

    def test_retrieve_params(self):

        sum_node = SPNCondSumNode(
            [DummyNode(Scope([0])), DummyNode(Scope([0]))]
        )

        # number of child outputs not matching number of weights
        sum_node.set_cond_f(lambda data: {"weights": [1.0]})
        self.assertRaises(
            ValueError,
            sum_node.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # non-positive weights
        sum_node.set_cond_f(lambda data: {"weights": [0.0, 1.0]})
        self.assertRaises(
            ValueError,
            sum_node.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # weights not summing up to one
        sum_node.set_cond_f(lambda data: {"weights": [0.5, 0.3]})
        self.assertRaises(
            ValueError,
            sum_node.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # weights of invalid shape
        sum_node.set_cond_f(lambda data: {"weights": [[0.5, 0.5]]})
        self.assertRaises(
            ValueError,
            sum_node.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # weights as list of floats
        sum_node.set_cond_f(lambda data: {"weights": [0.5, 0.5]})
        self.assertTrue(
            np.all(
                sum_node.retrieve_params(np.array([[1.0]]), DispatchContext())
                == np.array([0.5, 0.5])
            )
        )
        # weights as numpy array
        sum_node.set_cond_f(lambda data: {"weights": np.array([0.5, 0.5])})
        self.assertTrue(
            np.all(
                sum_node.retrieve_params(np.array([[1.0]]), DispatchContext())
                == np.array([0.5, 0.5])
            )
        )

    def test_marginalization_1(self):

        s = SPNCondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, s.scopes_out)

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg, None)

    def test_marginalization_2(self):

        s = SPNCondSumNode(
            [
                SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
                SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
            ]
        )

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg.scopes_out, [Scope([1])])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, [Scope([0])])

        s_marg = marginalize(s, [0, 1])
        self.assertEqual(s_marg, None)


if __name__ == "__main__":
    unittest.main()
