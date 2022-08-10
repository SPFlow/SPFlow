from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, marginalize
from spflow.meta.scope.scope import Scope
from ..nodes.dummy_node import DummyNode
import torch
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_sum_layer_initialization(self):

        # dummy children over same scope
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0,1]))]

        # ----- check attributes after correct initialization -----

        l = SPNSumLayer(n=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1]), Scope([0,1]), Scope([0,1])]))
        # make sure weight property works correctly
        self.assertTrue(l.weights.shape == (3, 3))

        # ----- same weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4]])

        # two dimensional weight array
        l = SPNSumLayer(n=3, children=input_nodes, weights=weights)

        for i in range(3):
            self.assertTrue(torch.all(l.weights[i] == weights))

        # one dimensional weight array
        l = SPNSumLayer(n=3, children=input_nodes, weights=weights.squeeze(0))

        for i in range(3):
            self.assertTrue(torch.all(l.weights[i] == weights))

        # ----- different weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])
        
        l = SPNSumLayer(n=3, children=input_nodes, weights=weights)
        for i in range(3):
            self.assertTrue(torch.all(l.weights[i] == weights[i]))

        # ----- two dimensional weight array of wrong shape -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights.T)

        # ----- weights not summing up to one per row -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)

        # ----- non-positive weights -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)

        # ----- children of different scopes -----
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0]))]
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes)
    
        # ----- no children -----
        self.assertRaises(ValueError, SPNSumLayer, 3, [])

    def test_sum_layer_structural_marginalization(self):
        
        # dummy children over same scope
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0,1]))]
        l = SPNSumLayer(n=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [0])
        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1]), Scope([0,1]), Scope([0,1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))
    
    def test_sum_layer_backend_conversion(self):
        #raise NotImplementedError()
        pass

    def test_product_layer_initialization(self):
        
        # dummy children pair-wise disjoint scopes
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([2]))]

        # ----- check attributes after correct initialization -----

        l = SPNProductLayer(n=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1,2,3]), Scope([0,1,2,3]), Scope([0,1,2,3])]))
        
        # ----- children of non-pair-wise disjoint scopes -----
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([1]))]
        self.assertRaises(ValueError, SPNProductLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, SPNProductLayer, 3, [])

    def test_product_layer_structural_marginalization(self):
        
        # dummy children over pair-wise disjoint scopes
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([2]))]
        l = SPNProductLayer(n=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1,2,3]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1,2]), Scope([0,1,2]), Scope([0,1,2])])
        # number of children should be reduced by one (i.e., marginalized over)
        self.assertTrue(len(list(l_marg.children())) == 2)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1,2,3]), Scope([0,1,2,3]), Scope([0,1,2,3])])

    def test_product_layer_backend_conversion(self):
        #raise NotImplementedError()
        pass

if __name__ == "__main__":
    unittest.main()