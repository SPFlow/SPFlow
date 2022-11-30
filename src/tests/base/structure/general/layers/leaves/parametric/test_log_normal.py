from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.base.structure.spn import LogNormal, LogNormalLayer, marginalize
from spflow.base.structure import AutoLeaf
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = LogNormalLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )
        # make sure parameter properties works correctly
        mean_values = l.mean
        std_values = l.std
        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.std == node_std))

        # ----- float/int parameter values -----
        mean_value = 2
        std_value = 0.5
        l = LogNormalLayer(
            scope=Scope([1]), n_nodes=3, mean=mean_value, std=std_value
        )

        for node in l.nodes:
            self.assertTrue(np.all(node.mean == mean_value))
            self.assertTrue(np.all(node.std == std_value))

        # ----- list parameter values -----
        mean_values = [1.0, 5.0, -3.0]
        std_values = [0.25, 0.5, 0.3]
        l = LogNormalLayer(
            scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values
        )

        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.std == node_std))

        # wrong number of values
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values[:-1],
            std_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values,
            std_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values,
            [std_values for _ in range(3)],
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            [mean_values for _ in range(3)],
            std_values,
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = LogNormalLayer(
            scope=Scope([1]),
            n_nodes=3,
            mean=np.array(mean_values),
            std=np.array(std_values),
        )

        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.std == node_std))

        # wrong number of values
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array(mean_values[:-1]),
            np.array(std_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array(mean_values),
            np.array(std_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values,
            np.array([std_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array([mean_values for _ in range(3)]),
            std_values,
            n_nodes=3,
        )

        # ---- different scopes -----
        l = LogNormalLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, LogNormalLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, LogNormalLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, LogNormalLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = LogNormalLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            LogNormalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            LogNormalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.LogNormal]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            LogNormalLayer.accepts(
                [
                    FeatureContext(
                        Scope([0]), [FeatureTypes.LogNormal(0.0, 1.0)]
                    ),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            LogNormalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(
            LogNormalLayer.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            LogNormalLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        log_normal = LogNormalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

        log_normal = LogNormalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.LogNormal]),
                FeatureContext(Scope([1]), [FeatureTypes.LogNormal]),
            ]
        )
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

        log_normal = LogNormalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.LogNormal(0.0, 1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.LogNormal(0.0, 1.0)]),
            ]
        )
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])
        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            LogNormalLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            LogNormalLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            LogNormalLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(LogNormalLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            LogNormalLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.LogNormal]),
                    FeatureContext(Scope([1]), [FeatureTypes.LogNormal]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.LogNormal(mean=-1.0, std=1.5)]
                ),
                FeatureContext(
                    Scope([1]), [FeatureTypes.LogNormal(mean=1.0, std=0.5)]
                ),
            ]
        )
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = LogNormalLayer(
            scope=Scope([1]), mean=[-0.2, 1.3], std=[0.5, 0.3], n_nodes=2
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(np.all(l.mean == l_marg.mean))
        self.assertTrue(np.all(l.std == l_marg.std))

        # ---------- different scopes -----------

        l = LogNormalLayer(
            scope=[Scope([1]), Scope([0])], mean=[-0.2, 1.3], std=[0.5, 0.3]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, LogNormal))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.mean, np.array([1.3]))
        self.assertEqual(l_marg.std, np.array([0.3]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, LogNormalLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.mean, np.array([1.3]))
        self.assertEqual(l_marg.std, np.array([0.3]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(np.all(l.mean == l_marg.mean))
        self.assertTrue(np.all(l.std == l_marg.std))

    def test_get_params(self):

        layer = LogNormalLayer(
            scope=Scope([1]), mean=[-0.73, 0.29], std=[1.3, 0.92], n_nodes=2
        )

        mean, std, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(mean, np.array([-0.73, 0.29])))
        self.assertTrue(np.allclose(std, np.array([1.3, 0.92])))


if __name__ == "__main__":
    unittest.main()
