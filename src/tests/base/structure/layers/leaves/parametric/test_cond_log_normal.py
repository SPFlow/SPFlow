from spflow.base.structure.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer, marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import CondLogNormal
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondLogNormalLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))

        # ---- different scopes -----
        l = CondLogNormalLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondLogNormalLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondLogNormalLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondLogNormalLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = CondLogNormalLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondLogNormalLayer(Scope([0]), n_nodes=2, cond_f=[lambda data: {'mean': 0.0, 'std': 1.0}, lambda data: {'mean': 0.0, 'std': 1.0}])
        self.assertRaises(ValueError, CondLogNormalLayer, Scope([0]), n_nodes=2, cond_f=[lambda data: {'mean': 0.0, 'std': 1.0}])

    def test_retrieve_params(self):

        # ----- float/int parameter values ----- 
        mean_value = 2
        std_value = 0.5
        l = CondLogNormalLayer(scope=Scope([1]), n_nodes=3, cond_f=lambda data: {'mean': mean_value, 'std': std_value})

        for mean_node, std_node in zip(*l.retrieve_params(np.array([[1.0]]), DispatchContext())):
            self.assertTrue(mean_node == mean_value)
            self.assertTrue(std_node == std_value)

        # ----- list parameter values -----
        mean_values = [1.0, 5.0, -3.0]
        std_values = [0.25, 0.5, 0.3]
        l.set_cond_f(lambda data: {'mean': mean_values, 'std': std_values})

        for mean_actual, std_actual, mean_node, std_node in zip(mean_values, std_values, *l.retrieve_params(np.array([[1.0]]), DispatchContext())):
            self.assertTrue(mean_actual == mean_node)
            self.assertTrue(std_actual == std_node)
        
        # wrong number of values
        l.set_cond_f(lambda data: {'mean': mean_values[:-1], 'std': std_values})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())
        l.set_cond_f(lambda data: {'mean': mean_values, 'std': std_values[:-1]})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {'mean': [mean_values for _ in range(3)], 'std': std_values})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())
        l.set_cond_f(lambda data: {'mean': mean_values, 'std': [std_values for _ in range(3)]})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {'mean': np.array(mean_values), 'std': np.array(std_values)})
        for mean_actual, std_actual, mean_node, std_node in zip(mean_values, std_values, *l.retrieve_params(np.array([[1.0]]), DispatchContext())):
            self.assertTrue(mean_node == mean_actual)
            self.assertTrue(std_node == std_actual)
        
        # wrong number of values
        l.set_cond_f(lambda data: {'mean': np.array(mean_values[:-1]), 'std': np.array(std_values)})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())
        l.set_cond_f(lambda data: {'mean': np.array(mean_values), 'std': np.array(std_values[:-1])})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {'mean': np.array([mean_values for _ in range(3)]), 'std': np.array(std_values)})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())
        l.set_cond_f(lambda data: {'mean': np.array(mean_values), 'std': np.array([std_values for _ in range(3)])})
        self.assertRaises(ValueError, l.retrieve_params, np.array([[1]]), DispatchContext())

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondLogNormalLayer(scope=Scope([1]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    
        # ---------- different scopes -----------

        l = CondLogNormalLayer(scope=[Scope([1]), Scope([0])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondLogNormal))
        self.assertEqual(l_marg.scope, Scope([0]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondLogNormalLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])


if __name__ == "__main__":
    unittest.main()