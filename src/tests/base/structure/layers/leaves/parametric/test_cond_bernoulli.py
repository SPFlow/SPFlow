from spflow.base.structure.layers.leaves.parametric.cond_bernoulli import (
    CondBernoulliLayer,
    marginalize,
)
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.cond_bernoulli import (
    CondBernoulli,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization(self):

        # ----- check attributes after correct initialization -----

        l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out
                == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]
            )
        )

        # ---- different scopes -----
        l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondBernoulliLayer, Scope([0], [1]), n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondBernoulliLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondBernoulliLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondBernoulliLayer(
            scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3
        )
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondBernoulliLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondBernoulliLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        p_value = 0.13
        l = CondBernoulliLayer(
            scope=Scope([1], [2]), n_nodes=3, cond_f=lambda data: {"p": p_value}
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

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            CondBernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Bernoulli feature type class
        self.assertTrue(
            CondBernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            CondBernoulliLayer.accepts(
                [
                    FeatureContext(
                        Scope([0], [2]), [FeatureTypes.Bernoulli(0.5)]
                    ),
                    FeatureContext(
                        Scope([1], [3]), [FeatureTypes.Bernoulli(0.5)]
                    ),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondBernoulliLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondBernoulliLayer.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondBernoulliLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        bernoulli = CondBernoulliLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(
            bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

        bernoulli = CondBernoulliLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli]),
            ]
        )
        self.assertTrue(
            bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

        bernoulli = CondBernoulliLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0], [2]), [FeatureTypes.Bernoulli(p=0.75)]
                ),
                FeatureContext(
                    Scope([1], [3]), [FeatureTypes.Bernoulli(p=0.25)]
                ),
            ]
        )
        self.assertTrue(
            bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondBernoulliLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondBernoulliLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondBernoulliLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondBernoulliLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondBernoulliLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli()]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli()]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        bernoulli = AutoLeaf(
            [
                FeatureContext(
                    Scope([0], [2]), [FeatureTypes.Bernoulli(p=0.75)]
                ),
                FeatureContext(
                    Scope([1], [3]), [FeatureTypes.Bernoulli(p=0.25)]
                ),
            ]
        )
        self.assertTrue(
            bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondBernoulliLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondBernoulli))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondBernoulliLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
