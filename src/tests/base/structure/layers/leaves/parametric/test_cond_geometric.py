from spflow.base.structure.layers.leaves.parametric.cond_geometric import (
    CondGeometricLayer,
    marginalize,
)
from spflow.base.structure.nodes.leaves.parametric.cond_geometric import (
    CondGeometric,
)
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondGeometricLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )

        # ---- different scopes -----
        l = CondGeometricLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = CondGeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondGeometricLayer(
            Scope([0]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondGeometricLayer,
            Scope([0]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        p_value = 0.13
        l = CondGeometricLayer(
            scope=Scope([1]), n_nodes=3, cond_f=lambda data: {"p": p_value}
        )

        for p in l.retrieve_params(np.array([[1.0]]), DispatchContext()):
            self.assertTrue(p == p_value)

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l.set_cond_f(lambda data: {"p": p_values})

        for p_node, p_actual in zip(
            l.retrieve_params(np.array([[1.0]]), DispatchContext()), p_values
        ):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": p_values[:-1]})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"p": np.array(p_values)})
        for p_node, p_actual in zip(
            l.retrieve_params(np.array([[1.0]]), DispatchContext()), p_values
        ):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": np.array(p_values[:-1])})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": np.array([p_values for _ in range(3)])})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGeometricLayer(scope=Scope([1]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])

        # ---------- different scopes -----------

        l = CondGeometricLayer(scope=[Scope([1]), Scope([0])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGeometric))
        self.assertEqual(l_marg.scope, Scope([0]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGeometricLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])


if __name__ == "__main__":
    unittest.main()
