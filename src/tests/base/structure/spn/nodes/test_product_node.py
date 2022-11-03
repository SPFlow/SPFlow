from spflow.base.structure.spn.nodes.product_node import (
    SPNProductNode,
    marginalize,
)
from spflow.meta.data.scope import Scope
from .dummy_node import DummyNode
import unittest


class TestProductNode(unittest.TestCase):
    def test_product_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, SPNProductNode, [])
        # non-Module children
        self.assertRaises(
            ValueError, SPNProductNode, [DummyNode(Scope([0])), 0]
        )
        # children with non-disjoint scopes
        self.assertRaises(
            ValueError,
            SPNProductNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
        )

        # correct initialization
        SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))])

    def test_product_node_marginalization_1(self):

        p = SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))])

        p_marg = marginalize(p, [2])
        self.assertEqual(p_marg.scopes_out, p.scopes_out)

        p_marg = marginalize(p, [1], prune=False)
        self.assertEqual(p_marg.scopes_out, [Scope([0])])

        p_marg = marginalize(p, [1], prune=True)
        # pruning should return single child directly
        self.assertTrue(isinstance(p_marg, DummyNode))

    def test_product_node_marginalization_2(self):

        p = SPNProductNode(
            [
                SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
                SPNProductNode([DummyNode(Scope([2])), DummyNode(Scope([3]))]),
            ]
        )

        p_marg = marginalize(p, [2])
        self.assertEqual(p_marg.scopes_out, [Scope([0, 1, 3])])


if __name__ == "__main__":
    unittest.main()
