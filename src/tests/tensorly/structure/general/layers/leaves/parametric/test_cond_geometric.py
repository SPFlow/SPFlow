import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import marginalize
from spflow.tensorly.structure.general.nodes.leaves import CondGeometric
from spflow.tensorly.structure.general.layers.leaves import CondGeometricLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

        # ---- different scopes -----
        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([0], [1]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondGeometricLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondGeometricLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        p_value = 0.13
        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value})

        for p in l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()):
            self.assertTrue(p == p_value)

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l.set_cond_f(lambda data: {"p": p_values})

        for p_node, p_actual in zip(l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()), p_values):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": p_values[:-1]})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"p": tl.tensor(p_values)})
        for p_node, p_actual in zip(l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()), p_values):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": tl.tensor(p_values[:-1])})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": tl.tensor([p_values for _ in range(3)])})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            CondGeometricLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondGeometricLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGeometricLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(CondGeometricLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Geometric])]))

        # multivariate signature
        self.assertFalse(
            CondGeometricLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Geometric, FeatureTypes.Geometric],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        geometric = CondGeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        geometric = CondGeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        geometric = CondGeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric(0.5)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric(0.5)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGeometricLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGeometricLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGeometricLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGeometricLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric(p=0.75)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric(p=0.25)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGeometric))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGeometricLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
