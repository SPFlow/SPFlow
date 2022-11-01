from spflow.base.structure.layers.leaves.parametric.cond_gamma import (
    CondGammaLayer,
    marginalize,
)
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=3)
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
        l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondGammaLayer, Scope([0], [1]), n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGammaLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGammaLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondGammaLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondGammaLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[
                lambda data: {"alpha": 0.5, "beta": 0.5},
                lambda data: {"alpha": 0.5, "beta": 0.5},
            ],
        )
        self.assertRaises(
            ValueError,
            CondGammaLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"alpha": 0.5, "beta": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        alpha_value = 2
        beta_value = 0.5
        l = CondGammaLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            cond_f=lambda data: {"alpha": alpha_value, "beta": beta_value},
        )

        for alpha_node, beta_node in zip(
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(alpha_node == alpha_value)
            self.assertTrue(beta_node == beta_value)

        # ----- list parameter values -----
        alpha_values = [1.0, 5.0, 3.0]
        beta_values = [0.25, 0.5, 0.3]
        l.set_cond_f(lambda data: {"alpha": alpha_values, "beta": beta_values})

        for alpha_actual, beta_actual, alpha_node, beta_node in zip(
            alpha_values,
            beta_values,
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(alpha_actual == alpha_node)
            self.assertTrue(beta_actual == beta_node)

        # wrong number of values
        l.set_cond_f(
            lambda data: {"alpha": alpha_values[:-1], "beta": beta_values}
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )
        l.set_cond_f(
            lambda data: {"alpha": alpha_values, "beta": beta_values[:-1]}
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "alpha": [alpha_values for _ in range(3)],
                "beta": beta_values,
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )
        l.set_cond_f(
            lambda data: {
                "alpha": alpha_values,
                "beta": [beta_values for _ in range(3)],
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # ----- numpy parameter values -----
        l.set_cond_f(
            lambda data: {
                "alpha": np.array(alpha_values),
                "beta": np.array(beta_values),
            }
        )
        for alpha_actual, beta_actual, alpha_node, beta_node in zip(
            alpha_values,
            beta_values,
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(alpha_node == alpha_actual)
            self.assertTrue(beta_node == beta_actual)

        # wrong number of values
        l.set_cond_f(
            lambda data: {
                "alpha": np.array(alpha_values[:-1]),
                "beta": np.array(beta_values),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )
        l.set_cond_f(
            lambda data: {
                "alpha": np.array(alpha_values),
                "beta": np.array(beta_values[:-1]),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "alpha": np.array([alpha_values for _ in range(3)]),
                "beta": np.array(beta_values),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )
        l.set_cond_f(
            lambda data: {
                "alpha": np.array(alpha_values),
                "beta": np.array([beta_values for _ in range(3)]),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGammaLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            CondGammaLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondGammaLayer.accepts(
                [
                    FeatureContext(
                        Scope([0], [2]), [FeatureTypes.Gamma(1.0, 1.0)]
                    ),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGammaLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondGammaLayer.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondGammaLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gamma = CondGammaLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
        self.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gamma = CondGammaLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma]),
            ]
        )
        self.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gamma = CondGammaLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma(1.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma(1.0, 1.0)]),
            ]
        )
        self.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGammaLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGammaLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGammaLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGammaLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGammaLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gamma = AutoLeaf(
            [
                FeatureContext(
                    Scope([0], [2]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)]
                ),
                FeatureContext(
                    Scope([1], [2]), [FeatureTypes.Gamma(alpha=0.5, beta=1.5)]
                ),
            ]
        )
        self.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondGammaLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGamma))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGammaLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
