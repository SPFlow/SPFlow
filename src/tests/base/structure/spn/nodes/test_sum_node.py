from spflow.base.structure.spn.nodes.sum_node import (
    SPNSumNode,
    marginalize,
)
from spflow.base.structure.spn.nodes.product_node import (
    SPNProductNode,
    marginalize,
)
from spflow.meta.data.scope import Scope
from .dummy_node import DummyNode
import numpy as np
import unittest


class TestSumNode(unittest.TestCase):
    def test_sum_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, SPNSumNode, [], [])
        # non-Module children
        self.assertRaises(
            ValueError, SPNSumNode, [DummyNode(Scope([0])), 0], [0.5, 0.5]
        )
        # children with different scopes
        self.assertRaises(
            ValueError,
            SPNSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([1]))],
            [0.5, 0.5],
        )
        # number of child outputs not matching number of weights
        self.assertRaises(
            ValueError,
            SPNSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [1.0],
        )
        # non-positive weights
        self.assertRaises(
            ValueError, SPNSumNode, [DummyNode(Scope([0]))], [0.0]
        )
        # weights not summing up to one
        self.assertRaises(
            ValueError,
            SPNSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [0.3, 0.5],
        )
        # weights of invalid shape
        self.assertRaises(
            ValueError, SPNSumNode, [DummyNode(Scope([0]))], [[1.0]]
        )

        # weights as list of floats
        SPNSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))], [0.5, 0.5])
        # weights as numpy array
        SPNSumNode(
            [DummyNode(Scope([0])), DummyNode(Scope([0]))], np.array([0.5, 0.5])
        )
        # no weights
        SPNSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

    def test_sum_node_marginalization_1(self):

        s = SPNSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, s.scopes_out)

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg, None)

    def test_sum_node_marginalization_2(self):

        s = SPNSumNode(
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
