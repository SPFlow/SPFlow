from spflow.base.structure.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer,
    marginalize,
)
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondBinomialLayer(scope=Scope([1]), n_nodes=3, n=2)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )

        # ----- n initialization -----
        l = CondBinomialLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])], n=[3, 5, 2]
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n=[3, 5],
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n=[[3, 5, 2]],
        )

        # n numpy array
        l = CondBinomialLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])], n=np.array([3, 5, 2])
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n=np.array([3, 5]),
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n=np.array([[3, 5, 2]]),
        )

        # ---- different scopes -----
        l = CondBinomialLayer(
            scope=[Scope([0]), Scope([1]), Scope([2])], n=5, n_nodes=3
        )
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.n, 5)
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondBinomialLayer, Scope([0]), 2, n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, CondBinomialLayer, Scope([]), 2, n_nodes=3
        )
        self.assertRaises(ValueError, CondBinomialLayer, [], n=2, n_nodes=3)

        # ----- invalid values for 'n' over same scope -----
        self.assertRaises(
            ValueError, CondBinomialLayer, Scope([0]), n=[2, 5], n_nodes=2
        )

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = CondBinomialLayer(scope=[Scope([1]), Scope([0])], n=2, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondBinomialLayer(
            Scope([0]),
            n=3,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            Scope([0]),
            n=3,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        n_value = 2
        p_value = 0.5
        l = CondBinomialLayer(
            scope=Scope([1]),
            n_nodes=3,
            n=n_value,
            cond_f=lambda data: {"p": p_value},
        )

        for p in l.retrieve_params(np.array([[1.0]]), DispatchContext()):
            self.assertTrue(p == p_value)

        # ----- list parameter values -----
        n_values = [1, 5, 4]
        p_values = [0.25, 0.5, 0.3]

        l = CondBinomialLayer(
            scope=[Scope([0]), Scope([1]), Scope([2])],
            n_nodes=3,
            n=n_values,
            cond_f=lambda data: {"p": p_values},
        )

        for n_actual, p_actual, n_node, p_node in zip(
            n_values,
            p_values,
            l.n,
            l.retrieve_params(np.array([[1.0]]), DispatchContext()),
        ):
            self.assertTrue(n_node == n_actual)
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

        # wrong shape
        l.set_cond_f(lambda data: {"p": np.array([p_values for _ in range(3)])})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 0)})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 1)})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondBinomialLayer(scope=Scope([1]), n=2, n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.n == l_marg.n))

        # ---------- different scopes -----------

        l = CondBinomialLayer(scope=[Scope([1]), Scope([0])], n=[2, 6])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondBinomial))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.n, np.array([6]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondBinomialLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.n, np.array([6]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.n == l_marg.n))

    def test_get_params(self):

        l = CondBinomialLayer(scope=Scope([1]), n=[2, 2], n_nodes=2)

        n, *others = l.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(n, np.array([2, 2])))


if __name__ == "__main__":
    unittest.main()
