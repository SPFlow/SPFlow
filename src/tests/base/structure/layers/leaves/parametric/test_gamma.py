from spflow.base.structure.layers.leaves.parametric.gamma import GammaLayer, marginalize
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = GammaLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        alpha_values = l.alpha
        beta_values = l.beta
        for node, node_alpha, node_beta in zip(l.nodes, alpha_values, beta_values):
            self.assertTrue(np.all(node.alpha == node_alpha))
            self.assertTrue(np.all(node.beta == node_beta))

        # ----- float/int parameter values ----- 
        alpha_value = 2
        beta_value = 0.5
        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_value, beta=beta_value)

        for node in l.nodes:
            self.assertTrue(np.all(node.alpha == alpha_value))
            self.assertTrue(np.all(node.beta == beta_value))

        # ----- list parameter values -----
        alpha_values = [1.0, 5.0, 3.0]
        beta_values = [0.25, 0.5, 0.3]
        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_values, beta=beta_values)

        for node, node_alpha, node_beta in zip(l.nodes, alpha_values, beta_values):
            self.assertTrue(np.all(node.alpha == node_alpha))
            self.assertTrue(np.all(node.beta == node_beta))
        
        # wrong number of values
        self.assertRaises(ValueError, GammaLayer, Scope([0]), alpha_values, beta_values[:-1], n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), alpha_values[:-1], beta_values, n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), alpha_values, [beta_values for _ in range(3)], n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), [alpha_values for _ in range(3)], beta_values, n_nodes=3)

        # ----- numpy parameter values -----

        l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=np.array(alpha_values), beta=np.array(beta_values))

        for node, node_alpha, node_beta in zip(l.nodes, alpha_values, beta_values):
            self.assertTrue(np.all(node.alpha == node_alpha))
            self.assertTrue(np.all(node.beta == node_beta))
        
        # wrong number of values
        self.assertRaises(ValueError, GammaLayer, Scope([0]), np.array(alpha_values), np.array(beta_values[:-1]), n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), np.array(alpha_values[:-1]), np.array(beta_values), n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), np.array([alpha_values for _ in range(3)]), beta_values, n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, Scope([0]), alpha_values, np.array([beta_values for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = GammaLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GammaLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GammaLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, GammaLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GammaLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GammaLayer(scope=Scope([1]), alpha=[0.2, 1.3], beta=[0.5, 0.3], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.alpha == l_marg.alpha))
        self.assertTrue(np.all(l.beta == l_marg.beta))
    
        # ---------- different scopes -----------

        l = GammaLayer(scope=[Scope([1]), Scope([0])], alpha=[0.2, 1.3], beta=[0.5, 0.3])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Gamma))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.alpha, np.array([1.3]))
        self.assertEqual(l_marg.beta, np.array([0.3]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GammaLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.alpha, np.array([1.3]))
        self.assertEqual(l_marg.beta, np.array([0.3]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.alpha == l_marg.alpha))
        self.assertTrue(np.all(l.beta == l_marg.beta))

    def test_get_params(self):

        layer = GammaLayer(scope=Scope([1]), alpha=[0.73, 0.29], beta=[1.3, 0.92], n_nodes=2)

        alpha, beta, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(alpha, np.array([0.73, 0.29])))
        self.assertTrue(np.allclose(beta, np.array([1.3, 0.92])))


if __name__ == "__main__":
    unittest.main()