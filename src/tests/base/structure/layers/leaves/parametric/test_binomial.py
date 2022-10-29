from spflow.base.structure.layers.leaves.parametric.binomial import (
    BinomialLayer,
    marginalize,
)
from spflow.base.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = BinomialLayer(scope=Scope([1]), n_nodes=3, n=2)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )
        # make sure parameter properties works correctly
        n_values = l.n
        p_values = l.p
        for node, node_n, node_p in zip(l.nodes, n_values, p_values):
            self.assertTrue(np.all(node.n == node_n))
            self.assertTrue(np.all(node.p == node_p))

        # ----- float/int parameter values -----
        n_value = 2
        p_value = 0.5
        l = BinomialLayer(scope=Scope([1]), n_nodes=3, n=n_value, p=p_value)

        for node in l.nodes:
            self.assertTrue(np.all(node.n == n_value))
            self.assertTrue(np.all(node.p == p_value))

        # ----- list parameter values -----
        n_values = [1, 5, 4]
        p_values = [0.25, 0.5, 0.3]
        l = BinomialLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])], n=n_values, p=p_values
        )

        for node, node_n, node_p in zip(l.nodes, n_values, p_values):
            self.assertTrue(np.all(node.n == node_n))
            self.assertTrue(np.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            n_values[:-1],
            p_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            n_values,
            p_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            [n_values for _ in range(3)],
            [p_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = BinomialLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n=np.array(n_values),
            p=np.array(p_values),
        )

        for node, node_n, node_p in zip(l.nodes, n_values, p_values):
            self.assertTrue(np.all(node.n == node_n))
            self.assertTrue(np.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.array(n_values[:-1]),
            np.array(p_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.array(n_values),
            np.array(p_values[:-1]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            n_values,
            np.array([p_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.array([n_values for _ in range(3)]),
            p_values,
            n_nodes=3,
        )

        # wrong shape
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.expand_dims(np.array(n_values), 0),
            np.array(p_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.expand_dims(np.array(n_values), 1),
            np.array(p_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.array(n_values),
            np.expand_dims(np.array(p_values), 0),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BinomialLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            np.array(n_values),
            np.expand_dims(np.array(p_values), 1),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = BinomialLayer(scope=Scope([1]), n=5, n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, BinomialLayer, Scope([0]), 2, n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, BinomialLayer, Scope([]), 2, n_nodes=3)
        self.assertRaises(ValueError, BinomialLayer, [], n=2, n_nodes=3)

        # ----- invalid values for 'n' over same scope -----
        self.assertRaises(
            ValueError, BinomialLayer, Scope([0]), n=[2, 5], n_nodes=2
        )

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = BinomialLayer(scope=[Scope([1]), Scope([0])], n=2, p=0.5, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = BinomialLayer(scope=Scope([1]), n=2, p=[0.5, 0.3], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.n == l_marg.n))
        self.assertTrue(np.all(l.p == l_marg.p))

        # ---------- different scopes -----------

        l = BinomialLayer(
            scope=[Scope([1]), Scope([0])], n=[2, 6], p=[0.5, 0.3]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Binomial))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.n, np.array([6]))
        self.assertEqual(l_marg.p, np.array([0.3]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, BinomialLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.n, np.array([6]))
        self.assertEqual(l_marg.p, np.array([0.3]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.n == l_marg.n))
        self.assertTrue(np.all(l.p == l_marg.p))

    def test_get_params(self):

        l = BinomialLayer(scope=Scope([1]), n=[2, 2], p=[0.73, 0.29], n_nodes=2)

        n, p, *others = l.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(n, np.array([2, 2])))
        self.assertTrue(np.allclose(p, np.array([0.73, 0.29])))


if __name__ == "__main__":
    unittest.main()
