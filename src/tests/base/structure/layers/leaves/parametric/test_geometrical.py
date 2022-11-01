from spflow.base.structure.layers.leaves.parametric.geometric import (
    GeometricLayer,
    marginalize,
)
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = GeometricLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )
        # make sure parameter properties works correctly
        p_values = l.p
        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(np.all(node.p == node_p))

        # ----- float/int parameter values -----
        p_value = 0.13
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_value)

        for node in l.nodes:
            self.assertTrue(np.all(node.p == p_value))

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)

        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(np.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(
            ValueError, GeometricLayer, Scope([0]), p_values[:-1], n_nodes=3
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            [p_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=np.array(p_values))

        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(np.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            np.array(p_values[:-1]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            np.array([p_values for _ in range(3)]),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = GeometricLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, GeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(
            GeometricLayer.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])]
            )
        )

        # multivariate signature
        self.assertFalse(
            GeometricLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Geometric, FeatureTypes.Geometric],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric(0.5)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(GeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            GeometricLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric(p=0.25)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GeometricLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.p == l_marg.p))

        # ---------- different scopes -----------

        l = GeometricLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Geometric))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.p, np.array([0.29]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GeometricLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.p, np.array([0.29]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.p == l_marg.p))

    def test_get_params(self):

        layer = GeometricLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        p, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(p, np.array([0.73, 0.29])))


if __name__ == "__main__":
    unittest.main()
