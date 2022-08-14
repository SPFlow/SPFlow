from spflow.base.structure.layers.leaves.parametric.uniform import UniformLayer, marginalize
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = UniformLayer(scope=Scope([1]), n_nodes=3, start=0.0, end=1.0)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        start = l.start
        for node, node_start in zip(l.nodes, start):
            self.assertTrue(np.all(node.start == node_start))
        end = l.end
        for node, node_end in zip(l.nodes, end):
            self.assertTrue(np.all(node.end == node_end))

        # ----- float/int parameter values -----
        start = -1.0
        end = 2

        l = UniformLayer(scope=Scope([1]), n_nodes=3, start=start, end=end)

        for node in l.nodes:
            self.assertTrue(np.all(node.start == start))
            self.assertTrue(np.all(node.end == end))

        # ----- list parameter values -----
        start = [-1.0, -2.0, 3.0]
        end = [2.0, -1.5, 4.0]

        l = UniformLayer(scope=Scope([1]), n_nodes=3, start=start, end=end)

        for node, node_start, node_end in zip(l.nodes, start, end):
            self.assertTrue(np.all(node.start == node_start))
            self.assertTrue(np.all(node.end == node_end))
        
        # wrong number of values
        self.assertRaises(ValueError, UniformLayer, Scope([0]), start, end[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, UniformLayer, Scope([0]), [start for _ in range(3)], [end for _ in range(3)], n_nodes=3)

        # ----- numpy parameter values -----

        l = UniformLayer(scope=Scope([1]), n_nodes=3, start=np.array(start), end=np.array(end))

        for node, node_start, node_end in zip(l.nodes, start, end):
            self.assertTrue(np.all(node.start == node_start))
            self.assertTrue(np.all(node.end == node_end))
        
        # wrong number of values
        self.assertRaises(ValueError, UniformLayer, Scope([0]), np.array(start), np.array(end[:-1]), n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, UniformLayer, Scope([0]), np.array([start for _ in range(3)]), np.array([end for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = UniformLayer(scope=Scope([1]), start=0.0, end=1.0, n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, UniformLayer, Scope([0]), 0.0, 1.0, n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, UniformLayer, Scope([]), 0.0, 1.0, n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = UniformLayer(scope=[Scope([1]), Scope([0])], start=0.0, end=1.0, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = UniformLayer(scope=Scope([1]), start=[-1.0, 2.0], end=[1.0, 2.5], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.start == l_marg.start))
        self.assertTrue(np.all(l.end == l_marg.end))
    
        # ---------- different scopes -----------

        l = UniformLayer(scope=[Scope([1]), Scope([0])], start=[-1.0, 2.0], end=[1.0, 2.5])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Uniform))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.start, 2.0)
        self.assertEqual(l_marg.end, 2.5)

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, UniformLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.start, np.array([2.0]))
        self.assertEqual(l_marg.end, np.array([2.5]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.start == l_marg.start))
        self.assertTrue(np.all(l.end == l_marg.end))


if __name__ == "__main__":
    unittest.main()