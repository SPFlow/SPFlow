from spflow.base.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer, marginalize
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=1, M=1, n=1)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        N_values = l.N
        M_values = l.M
        n_values = l.n
        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(np.all(node.N == node_N))
            self.assertTrue(np.all(node.M == node_M))
            self.assertTrue(np.all(node.n == node_n))

        # ----- float/int parameter values ----- 
        N_value = 5
        M_value = 3
        n_value = 2
        l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=N_value, M=M_value, n=n_value)

        for node in l.nodes:
            self.assertTrue(np.all(node.N == N_value))
            self.assertTrue(np.all(node.M == M_value))
            self.assertTrue(np.all(node.n == n_value))

        # ----- list parameter values -----
        N_values = [3, 5, 4]
        M_values = [2, 1, 3]
        n_values = [2, 2, 3]
        l = HypergeometricLayer(scope=[Scope([1]), Scope([0]), Scope([2])], n_nodes=3, N=N_values, M=M_values, n=n_values)

        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(np.all(node.N == node_N))
            self.assertTrue(np.all(node.M == node_M))
            self.assertTrue(np.all(node.n == node_n))

        # wrong number of values
        self.assertRaises(ValueError, HypergeometricLayer, [Scope([1]), Scope([0]), Scope([2])], N_values, M_values, n_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, HypergeometricLayer, [Scope([1]), Scope([0]), Scope([2])], [N_values for _ in range(3)], [M_values for _ in range(3)], [n_values for _ in range(3)], n_nodes=3)

        # ----- numpy parameter values -----

        l = HypergeometricLayer(scope=[Scope([1]), Scope([0]), Scope([2])], N=np.array(N_values), M=np.array(M_values), n=np.array(n_values))

        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(np.all(node.N == node_N))
            self.assertTrue(np.all(node.M == node_M))
            self.assertTrue(np.all(node.n == node_n))
        
        # wrong number of values
        self.assertRaises(ValueError, HypergeometricLayer, [Scope([1]), Scope([0]), Scope([2])], np.array(N_values), np.array(M_values[:-1]), np.array(n_values[:-1]), n_nodes=3)
        self.assertRaises(ValueError, HypergeometricLayer, [Scope([1]), Scope([0]), Scope([2])], np.array([N_values for _ in range(3)]), np.array([M_values for _ in range(3)]), np.array([n_values for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = HypergeometricLayer(scope=Scope([1]), N=1, M=1, n=1, n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, HypergeometricLayer, Scope([0]), N=1, M=1, n=1, n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, HypergeometricLayer, Scope([]), N=1, M=1, n=1, n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], N=1, M=1, n=1, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = HypergeometricLayer(scope=Scope([1]), N=4, M=2, n=3, n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.N == l_marg.N))
        self.assertTrue(np.all(l.M == l_marg.M))
        self.assertTrue(np.all(l.n == l_marg.n))
    
        # ---------- different scopes -----------

        l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], N=[2, 6], M=[2, 4], n=[2, 5])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Hypergeometric))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.N, np.array([6]))
        self.assertEqual(l_marg.M, np.array([4]))
        self.assertEqual(l_marg.n, np.array([5]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, HypergeometricLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.N, np.array([6]))
        self.assertEqual(l_marg.M, np.array([4]))
        self.assertEqual(l_marg.n, np.array([5]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.N == l_marg.N))
        self.assertTrue(np.all(l.M == l_marg.M))
        self.assertTrue(np.all(l.n == l_marg.n))


if __name__ == "__main__":
    unittest.main()