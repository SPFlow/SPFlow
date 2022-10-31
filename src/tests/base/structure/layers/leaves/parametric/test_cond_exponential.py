from spflow.base.structure.layers.leaves.parametric.cond_exponential import (
    CondExponentialLayer,
    marginalize,
)
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])])
        )

        # ---- different scopes -----
        l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondExponentialLayer, Scope([0], [1]), n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, CondExponentialLayer, Scope([]), n_nodes=3
        )
        self.assertRaises(ValueError, CondExponentialLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondExponentialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondExponentialLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"l": 1.5}, lambda data: {"l": 1.5}],
        )
        self.assertRaises(
            ValueError,
            CondExponentialLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"l": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        l_value = 2
        l = CondExponentialLayer(
            scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
        )

        for l_node in l.retrieve_params(np.array([[1.0]]), DispatchContext()):
            self.assertTrue(l_node == l_value)

        # ----- list parameter values -----
        l_values = [1.0, 2.0, 3.0]
        l.set_cond_f(lambda data: {"l": l_values})

        for l_node, l_actual in zip(
            l.retrieve_params(np.array([[1.0]]), DispatchContext()), l_values
        ):
            self.assertTrue(l_node == l_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"l": l_values[:-1]})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"l": [l_values for _ in range(3)]})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"l": np.array(l_values)})
        for l_node, l_actual in zip(
            l.retrieve_params(np.array([[1.0]]), DispatchContext()), l_values
        ):
            self.assertTrue(l_node == l_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"l": np.array(l_values[:-1])})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"l": np.array([l_values for _ in range(3)])})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )
    
    def test_accept(self):

        # continuous meta type
        self.assertTrue(CondExponentialLayer.accepts([([FeatureTypes.Continuous], Scope([0], [2])), ([FeatureTypes.Continuous], Scope([1], [2]))]))

        # Exponential feature type class
        self.assertTrue(CondExponentialLayer.accepts([([FeatureTypes.Exponential], Scope([0], [2])), ([FeatureTypes.Continuous], Scope([1], [2]))]))

        # Exponential feature type instance
        self.assertTrue(CondExponentialLayer.accepts([([FeatureTypes.Exponential(1.0)], Scope([0], [2])), ([FeatureTypes.Continuous], Scope([1], [2]))]))

        # invalid feature type
        self.assertFalse(CondExponentialLayer.accepts([([FeatureTypes.Discrete], Scope([0], [2])), ([FeatureTypes.Continuous], Scope([1], [2]))]))

        # non-conditional scope
        self.assertFalse(CondExponentialLayer.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # scope length does not match number of types
        self.assertFalse(CondExponentialLayer.accepts([([FeatureTypes.Continuous], Scope([0, 1], [2]))]))

        # multivariate signature
        self.assertFalse(CondExponentialLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))]))

    def test_initialization_from_signatures(self):

        exponential = CondExponentialLayer.from_signatures([([FeatureTypes.Continuous], Scope([0], [2])), ([FeatureTypes.Continuous], Scope([1], [2]))])
        self.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        exponential = CondExponentialLayer.from_signatures([([FeatureTypes.Exponential], Scope([0], [2])), ([FeatureTypes.Exponential], Scope([1], [2]))])
        self.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        exponential = CondExponentialLayer.from_signatures([([FeatureTypes.Exponential(l=1.5)], Scope([0], [2])), ([FeatureTypes.Exponential(l=0.5)], Scope([1], [2]))])
        self.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, CondExponentialLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # non-conditional scope
        self.assertRaises(ValueError, CondExponentialLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, CondExponentialLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1], [2]))])

        # multivariate signature
        self.assertRaises(ValueError, CondExponentialLayer.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondExponentialLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(CondExponentialLayer, AutoLeaf.infer([([FeatureTypes.Exponential], Scope([0], [2])), ([FeatureTypes.Exponential], Scope([1], [2]))]))

        # make sure AutoLeaf can return correctly instantiated object
        exponential = AutoLeaf([([FeatureTypes.Exponential(l=1.5)], Scope([0], [2])), ([FeatureTypes.Exponential(l=0.5)], Scope([1], [2]))])
        self.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondExponentialLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondExponential))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondExponentialLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
