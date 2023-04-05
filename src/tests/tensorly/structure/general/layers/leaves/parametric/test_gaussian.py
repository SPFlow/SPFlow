import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import marginalize
from spflow.tensorly.structure.general.nodes.leaves import Gaussian
from spflow.tensorly.structure.general.layers.leaves import GaussianLayer
from spflow.tensorly.utils.helper_functions import tl_allclose
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = GaussianLayer(scope=Scope([1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        mean_values = l.mean
        std_values = l.std
        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.std == node_std))

        # ----- float/int parameter values -----
        mean_value = 2
        std_value = 0.5
        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_value, std=std_value)

        for node in l.nodes:
            self.assertTrue(tl.all(node.mean == mean_value))
            self.assertTrue(tl.all(node.std == std_value))

        # ----- list parameter values -----
        mean_values = [1.0, 5.0, -3.0]
        std_values = [0.25, 0.5, 0.3]
        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values)

        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.std == node_std))

        # wrong number of values
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            mean_values[:-1],
            std_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            mean_values,
            std_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            mean_values,
            [std_values for _ in range(3)],
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            [mean_values for _ in range(3)],
            std_values,
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = GaussianLayer(
            scope=Scope([1]),
            n_nodes=3,
            mean=tl.tensor(mean_values),
            std=tl.tensor(std_values),
        )

        for node, node_mean, node_std in zip(l.nodes, mean_values, std_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.std == node_std))

        # wrong number of values
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            tl.tensor(mean_values[:-1]),
            tl.tensor(std_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            tl.tensor(mean_values),
            tl.tensor(std_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            mean_values,
            tl.tensor([std_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            GaussianLayer,
            Scope([0]),
            tl.tensor([mean_values for _ in range(3)]),
            std_values,
            n_nodes=3,
        )

        # ---- different scopes -----
        l = GaussianLayer(scope=Scope([1]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GaussianLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, GaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GaussianLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            GaussianLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            GaussianLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            GaussianLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            GaussianLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(GaussianLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            GaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gaussian = GaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

        gaussian = GaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1]), [FeatureTypes.Gaussian]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

        gaussian = GaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Gaussian(0.0, 1.0)]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            GaussianLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            GaussianLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            GaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(GaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            GaussianLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1]), [FeatureTypes.Gaussian]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian(mean=-1.0, std=1.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Gaussian(mean=1.0, std=0.5)]),
            ]
        )
        self.assertTrue(isinstance(gaussian, GaussianLayer))
        self.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GaussianLayer(scope=Scope([1]), mean=[-0.2, 1.3], std=[0.5, 0.3], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(tl.all(l.mean == l_marg.mean))
        self.assertTrue(tl.all(l.std == l_marg.std))

        # ---------- different scopes -----------

        l = GaussianLayer(scope=[Scope([1]), Scope([0])], mean=[-0.2, 1.3], std=[0.5, 0.3])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.mean, tl.tensor([1.3]))
        self.assertEqual(l_marg.std, tl.tensor([0.3]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GaussianLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.mean, tl.tensor([1.3]))
        self.assertEqual(l_marg.std, tl.tensor([0.3]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(tl.all(l.mean == l_marg.mean))
        self.assertTrue(tl.all(l.std == l_marg.std))

    def test_get_params(self):

        layer = GaussianLayer(scope=Scope([1]), mean=[-0.73, 0.29], std=[1.3, 0.92], n_nodes=2)

        mean, std, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(tl_allclose(mean, tl.tensor([-0.73, 0.29])))
        self.assertTrue(tl_allclose(std, tl.tensor([1.3, 0.92])))


if __name__ == "__main__":
    unittest.main()
