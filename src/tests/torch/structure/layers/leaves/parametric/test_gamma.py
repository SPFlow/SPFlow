from spflow.torch.structure.layers.leaves.parametric.gamma import GammaLayer, marginalize, toTorch, toBase
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.structure.layers.leaves.parametric.gamma import GammaLayer as BaseGammaLayer
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
        alpha_values = [0.5, 2.3, 1.0]
        beta_values = [1.3, 1.0, 0.2]
        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_values, beta=beta_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(l.alpha, l.beta, alpha_values, beta_values):
            self.assertTrue(torch.allclose(alpha_layer_node, torch.tensor(alpha_value)))
            self.assertTrue(torch.allclose(beta_layer_node, torch.tensor(beta_value)))

        # ----- float/int parameter values -----
        alpha_value = 0.73
        beta_value = 1.9
        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_value, beta=beta_value)

        for alpha_layer_node, beta_layer_node in zip(l.alpha, l.beta):
            self.assertTrue(torch.allclose(alpha_layer_node, torch.tensor(alpha_value)))
            self.assertTrue(torch.allclose(beta_layer_node, torch.tensor(beta_value)))

        # ----- list parameter values -----
        alpha_values = [0.17, 0.8, 0.53]
        beta_values = [0.9, 1.34, 0.98]
        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_values, beta=beta_values)

        for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(l.alpha, l.beta, alpha_values, beta_values):
            self.assertTrue(torch.allclose(alpha_layer_node, torch.tensor(alpha_value)))
            self.assertTrue(torch.allclose(beta_layer_node, torch.tensor(beta_value)))

        # wrong number of values
        self.assertRaises(ValueError, GammaLayer, Scope([0]), alpha_values, beta_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), [alpha_values for _ in range(3)], [beta_values for _ in range(3)], n_nodes=3)

        # ----- numpy parameter values -----

        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=np.array(alpha_values), beta=np.array(beta_values))

        for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(l.alpha, l.beta, alpha_values, beta_values):
            self.assertTrue(torch.allclose(alpha_layer_node, torch.tensor(alpha_value)))
            self.assertTrue(torch.allclose(beta_layer_node, torch.tensor(beta_value)))

        # wrong number of values
        self.assertRaises(ValueError, GammaLayer, Scope([0]), np.array(alpha_values), np.array(beta_values[:-1]), n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), np.array([alpha_values for _ in range(3)]), np.array([beta_values for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = GammaLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GammaLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GammaLayer, Scope([]), n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GammaLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        
        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GammaLayer(scope=Scope([1]), alpha=[0.73, 0.29], beta=[0.41, 1.9], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.alpha, l_marg.alpha))
        self.assertTrue(torch.allclose(l.beta, l_marg.beta))

        # ---------- different scopes -----------

        l = GammaLayer(scope=[Scope([1]), Scope([0])], alpha=[0.73, 0.29], beta=[0.41, 1.9])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Gamma))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.alpha, torch.tensor(0.29)))
        self.assertTrue(torch.allclose(l_marg.beta, torch.tensor(1.9)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GammaLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.alpha, torch.tensor(0.29)))
        self.assertTrue(torch.allclose(l_marg.beta, torch.tensor(1.9)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.alpha, l_marg.alpha))
        self.assertTrue(torch.allclose(l.beta, l_marg.beta))

    def test_layer_dist(self):

        alpha_values = [0.73, 0.29, 0.5]
        beta_values = [0.9, 1.34, 0.98]
        l = GammaLayer(scope=Scope([1]), alpha=alpha_values, beta=beta_values, n_nodes=3)

        # ----- full dist -----
        dist = l.dist()

        for alpha_value, beta_value, alpha_dist, beta_dist in zip(alpha_values, beta_values, dist.concentration, dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(alpha_value), alpha_dist))
            self.assertTrue(torch.allclose(torch.tensor(beta_value), beta_dist))
        
        # ----- partial dist -----
        dist = l.dist([1,2])

        for alpha_value, beta_value, alpha_dist, beta_dist in zip(alpha_values[1:], beta_values[1:], dist.concentration, dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(alpha_value), alpha_dist))
            self.assertTrue(torch.allclose(torch.tensor(beta_value), beta_dist))

        dist = l.dist([1,0])

        for alpha_value, beta_value, alpha_dist, beta_dist in zip(reversed(alpha_values[:-1]), reversed(beta_values[:-1]), dist.concentration, dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(alpha_value), alpha_dist))
            self.assertTrue(torch.allclose(torch.tensor(beta_value), beta_dist))

    def test_layer_backend_conversion_1(self):
        
        torch_layer = GammaLayer(scope=[Scope([0]), Scope([1]), Scope([0])], alpha=[0.2, 0.9, 0.31], beta=[1.9, 0.3, 0.71])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.alpha, torch_layer.alpha.detach().numpy()))
        self.assertTrue(np.allclose(base_layer.beta, torch_layer.beta.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)
    
    def test_layer_backend_conversion_2(self):

        base_layer = BaseGammaLayer(scope=[Scope([0]), Scope([1]), Scope([0])], alpha=[0.2, 0.9, 0.31], beta=[1.9, 0.3, 0.71])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.alpha, torch_layer.alpha.detach().numpy()))
        self.assertTrue(np.allclose(base_layer.beta, torch_layer.beta.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()