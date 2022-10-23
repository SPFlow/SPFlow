from spflow.torch.structure.layers.leaves.parametric.geometric import GeometricLayer, marginalize, toTorch, toBase
from spflow.torch.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.structure.layers.leaves.parametric.geometric import GeometricLayer as BaseGeometricLayer
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
        p_values = [0.5, 0.3, 0.9]
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- float/int parameter values -----
        p_value = 0.73
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_value)

        for p_layer_node in l.p:
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)

        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), p_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), [p_values for _ in range(3)], n_nodes=3)

        # ----- numpy parameter values -----

        p = GeometricLayer(scope=Scope([1]), n_nodes=3, p=np.array(p_values))

        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), np.array(p_values[:-1]), n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), np.array([p_values for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = GeometricLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, GeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        
        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GeometricLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.p, l_marg.p))
    
        # ---------- different scopes -----------

        l = GeometricLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Geometric))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GeometricLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.p, l_marg.p))

    def test_layer_dist(self):

        p_values = [0.73, 0.29, 0.5]
        l = GeometricLayer(scope=Scope([1]), p=p_values, n_nodes=3)

        # ----- full dist -----
        dist = l.dist()

        for p_value, p_dist in zip(p_values, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))
        
        # ----- partial dist -----
        dist = l.dist([1,2])

        for p_value, p_dist in zip(p_values[1:], dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))

        dist = l.dist([1,0])

        for p_value, p_dist in zip(reversed(p_values[:-1]), dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))

    def test_layer_backend_conversion_1(self):
        
        torch_layer = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)
    
    def test_layer_backend_conversion_2(self):

        base_layer = BaseGeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()