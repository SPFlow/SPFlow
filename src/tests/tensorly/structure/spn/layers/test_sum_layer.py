import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_squeeze, tl_unsqueeze

from spflow.tensorly.structure.spn import SumLayer, marginalize
from spflow.meta.data import Scope

from ...general.nodes.dummy_node import DummyNode


class TestLayer(unittest.TestCase):
    def test_sum_layer_initialization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]

        # ----- check attributes after correct initialization -----

        l = SumLayer(n_nodes=3, children=input_nodes)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]))
        # make sure weight property works correctly
        weights = l.weights
        for node, node_weights in zip(l.nodes, weights):
            self.assertTrue(tl.all(node.weights == node_weights))

        # ----- same weights for all nodes -----
        weights = tl.tensor([[0.3, 0.3, 0.4]])

        # two dimensional weight array
        l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)

        for node in l.nodes:
            self.assertTrue(tl.all(node.weights == weights))

        # one dimensional weight array
        l = SumLayer(n_nodes=3, children=input_nodes, weights=tl_squeeze(weights,0))

        for node in l.nodes:
            self.assertTrue(tl.all(node.weights == weights))

        # ----- different weights for all nodes -----
        weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])

        l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)

        for node, node_weights in zip(l.nodes, weights):
            self.assertTrue(tl.all(node.weights == node_weights))

        # ----- two dimensional weight array of wrong shape -----
        weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights.T)
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, tl_unsqueeze(weights, 0))
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, tl_unsqueeze(weights, -1))

        # ----- incorrect number of weights -----
        weights = tl.tensor([[0.3, 0.3, 0.3, 0.1], [0.5, 0.2, 0.2, 0.1]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        weights = tl.tensor([[0.3, 0.7], [0.5, 0.5]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        # ----- weights not summing up to one per row -----
        weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        # ----- non-positive weights -----
        weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        # ----- children of different scopes -----
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0])),
        ]
        self.assertRaises(ValueError, SumLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, SumLayer, 3, [])

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, SumLayer, 0, input_nodes)

    def test_sum_layer_structural_marginalization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]
        l = SumLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(
            l,
            [0],
        )
        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        self.assertTrue(tl.all(l.weights == l_marg.weights))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])])
        self.assertTrue(tl.all(l.weights == l_marg.weights))


if __name__ == "__main__":
    unittest.main()
