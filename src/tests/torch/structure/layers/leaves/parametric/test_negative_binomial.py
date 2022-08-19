from spflow.torch.structure.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer, marginalize, toTorch, toBase
from spflow.torch.structure.nodes.leaves.parametric.negative_binomial import NegativeBinomial
from spflow.base.structure.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as BaseNegativeBinomialLayer
from spflow.meta.scope.scope import Scope
import torch
import numpy as np
import unittest
import itertools


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):
        
        # ----- check attributes after correct initialization -----
        n_values = [3, 2, 7]
        p_values = [0.3, 0.7, 0.5]
        l = NegativeBinomialLayer(scope=[Scope([1]), Scope([0]), Scope([2])], n=n_values, p=p_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([0]), Scope([2])]))
        # make sure parameter properties works correctly
        for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- float/int parameter values -----
        n_value = 5
        p_value = 0.13
        l = NegativeBinomialLayer(scope=Scope([1]), n_nodes=3, n=n_value, p=p_value)

        for n_layer_node, p_layer_node in zip(l.n, l.p):
            self.assertTrue(torch.all(n_layer_node == n_value))
            self.assertTrue(torch.all(p_layer_node == p_value))

        # ----- list parameter values -----
        n_values = [3, 2, 7]
        p_values = [0.17, 0.8, 0.53]
        l = NegativeBinomialLayer(scope=[Scope([1]), Scope([0]), Scope([2])], n=n_values, p=p_values)

        for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], n_values[:-1], p_values)
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], n_values, p_values[:-1])
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], n_values, [p_values for _ in range(3)])
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], [n_values for _ in range(3)], p_values)

        # ----- numpy parameter values -----

        l = NegativeBinomialLayer(scope=[Scope([0]), Scope([1]), Scope([2])], n=np.array(n_values), p=np.array(p_values))

        for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], np.array(n_values[:-1]), np.array(p_values))
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], np.array(n_values), np.array(p_values[:-1]))
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], np.array(n_values), np.array([p_values for _ in range(3)]), n_nodes=3)
        self.assertRaises(ValueError, NegativeBinomialLayer, [Scope([0]), Scope([1]), Scope([2])], np.array([n_values for _ in range(3)]), np.array(p_values), n_nodes=3)

        # ---- different scopes -----
        l = NegativeBinomialLayer(scope=Scope([1]), n_nodes=3, n=2)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, NegativeBinomialLayer, Scope([0]), n_nodes=0, n=2)

        # ----- invalid scope -----
        self.assertRaises(ValueError, NegativeBinomialLayer, Scope([]), n_nodes=3, n=2)
        self.assertRaises(ValueError, NegativeBinomialLayer, [], n_nodes=3, n=2)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = NegativeBinomialLayer(scope=[Scope([1]), Scope([0])], n_nodes=3, n=2)
        
        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = NegativeBinomialLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2, n=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.p, l_marg.p))
    
        # ---------- different scopes -----------

        l = NegativeBinomialLayer(scope=[Scope([1]), Scope([0])], n=[3, 2], p=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, NegativeBinomial))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(2)))
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, NegativeBinomialLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(2)))
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.n, l_marg.n))
        self.assertTrue(torch.allclose(l.p, l_marg.p))
    
    def test_layer_dist(self):

        n_values = [3, 2, 7]
        p_values = [0.73, 0.29, 0.5]
        l = NegativeBinomialLayer(scope=[Scope([1]), Scope([0]), Scope([2])], n=n_values, p=p_values)

        # ----- full dist -----
        dist = l.dist()

        for n_value, p_value, n_dist, p_dist in zip(n_values, p_values, dist.total_count, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(n_value).double(), n_dist))
            self.assertTrue(torch.allclose(torch.tensor(1-p_value).double(), p_dist))
        
        # ----- partial dist -----
        dist = l.dist([1,2])

        for n_value, p_value, n_dist, p_dist in zip(n_values[1:], p_values[1:], dist.total_count, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(n_value).double(), n_dist))
            self.assertTrue(torch.allclose(torch.tensor(1-p_value).double(), p_dist))

        dist = l.dist([1,0])

        for n_value, p_value, n_dist, p_dist in zip(reversed(n_values[:-1]), reversed(p_values[:-1]), dist.total_count, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(n_value).double(), n_dist))
            self.assertTrue(torch.allclose(torch.tensor(1-p_value).double(), p_dist))

    def test_layer_backend_conversion_1(self):
        
        torch_layer = NegativeBinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], n=[2, 5, 2], p=[0.2, 0.9, 0.31])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)
    
    def test_layer_backend_conversion_2(self):

        base_layer = BaseNegativeBinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], n=[2, 5, 2], p=[0.2, 0.9, 0.31])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()