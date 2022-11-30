import unittest

import numpy as np

from spflow.base.structure.spn.layers.product_layer import ProductLayer, marginalize
from spflow.meta.data import Scope

from ...general.nodes.dummy_node import DummyNode


class TestLayer(unittest.TestCase):
    def test_product_layer_initialization(self):

        # dummy children pair-wise disjoint scopes
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([3])),
            DummyNode(Scope([2])),
        ]

        # ----- check attributes after correct initialization -----

        l = ProductLayer(n_nodes=3, children=input_nodes)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out
                == [
                    Scope([0, 1, 2, 3]),
                    Scope([0, 1, 2, 3]),
                    Scope([0, 1, 2, 3]),
                ]
            )
        )

        # ----- children of non-pair-wise disjoint scopes -----
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([3])),
            DummyNode(Scope([1])),
        ]
        self.assertRaises(ValueError, ProductLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, ProductLayer, 3, [])

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, ProductLayer, 0, input_nodes)

    def test_product_layer_structural_marginalization(self):

        # dummy children over pair-wise disjoint scopes
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([3])),
            DummyNode(Scope([2])),
        ]
        l = ProductLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [3])
        self.assertTrue(
            l_marg.scopes_out
            == [Scope([0, 1, 2]), Scope([0, 1, 2]), Scope([0, 1, 2])]
        )
        # number of children should be reduced by one (i.e., marginalized over)
        self.assertTrue(len(l_marg.children) == 2)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])
        self.assertTrue(
            l_marg.scopes_out
            == [Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3])]
        )

        # ----- pruning -----
        l = ProductLayer(n_nodes=3, children=input_nodes[:2])

        l_marg = marginalize(l, [0, 1], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))


if __name__ == "__main__":
    unittest.main()
