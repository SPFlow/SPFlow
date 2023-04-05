import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import marginalize
from spflow.tensorly.structure.general.nodes.leaves import Bernoulli
from spflow.tensorly.structure.general.layers.leaves import BernoulliLayer
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.meta.data import FeatureContext, FeatureTypes, Scope



class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = BernoulliLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        p_values = l.p
        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(tl.all(node.p == node_p))

        # ----- float/int parameter values -----
        p_value = 0.13
        l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=p_value)

        for node in l.nodes:
            self.assertTrue(tl.all(node.p == p_value))

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=p_values)

        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(tl.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(ValueError, BernoulliLayer, Scope([0]), p_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            BernoulliLayer,
            Scope([0]),
            [p_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=tl.tensor(p_values))

        for node, node_p in zip(l.nodes, p_values):
            self.assertTrue(tl.all(node.p == node_p))

        # wrong number of values
        self.assertRaises(
            ValueError,
            BernoulliLayer,
            Scope([0]),
            tl.tensor(p_values[:-1]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            BernoulliLayer,
            Scope([0]),
            tl.tensor([p_values for _ in range(3)]),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = BernoulliLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, BernoulliLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, BernoulliLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, BernoulliLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = BernoulliLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            BernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            BernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Bernoulli]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            BernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(0.5)]),
                    FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(0.5)]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            BernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(BernoulliLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            BernoulliLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        bernoulli = BernoulliLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(tl.all(bernoulli.p == tl.tensor([0.5, 0.5])))
        self.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

        bernoulli = BernoulliLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Bernoulli]),
                FeatureContext(Scope([1]), [FeatureTypes.Bernoulli]),
            ]
        )
        self.assertTrue(tl.all(bernoulli.p == tl.tensor([0.5, 0.5])))
        self.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

        bernoulli = BernoulliLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)]),
                FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(p=0.25)]),
            ]
        )
        self.assertTrue(tl.all(bernoulli.p == tl.tensor([0.75, 0.25])))
        self.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            BernoulliLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            BernoulliLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            BernoulliLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(BernoulliLayer))

        feature_ctx_1 = FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])
        feature_ctx_2 = FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(p=0.25)])

        # make sure leaf is correctly inferred
        self.assertEqual(BernoulliLayer, AutoLeaf.infer([feature_ctx_1, feature_ctx_2]))

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf([feature_ctx_1, feature_ctx_2])
        self.assertTrue(tl.all(bernoulli.p == tl.tensor([0.75, 0.25])))
        self.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = BernoulliLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(tl.all(l.p == l_marg.p))

        # ---------- different scopes -----------

        l = BernoulliLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Bernoulli))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.p, tl.tensor([0.29]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, BernoulliLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.p, tl.tensor([0.29]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(tl.all(l.p == l_marg.p))

    def test_get_params(self):

        l = BernoulliLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        p, *others = l.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(tl_allclose(p, tl.tensor([0.73, 0.29])))


if __name__ == "__main__":
    unittest.main()
