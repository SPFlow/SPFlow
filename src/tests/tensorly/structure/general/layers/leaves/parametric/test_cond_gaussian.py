import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import CondGaussian, CondGaussianLayer, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

        # ---- different scopes -----
        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondGaussianLayer, Scope([0], [1]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGaussianLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondGaussianLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[
                lambda data: {"mean": 0.0, "std": 1.0},
                lambda data: {"mean": 0.0, "std": 1.0},
            ],
        )
        self.assertRaises(
            ValueError,
            CondGaussianLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"mean": 0.0, "std": 1.0}],
        )

    def test_retrieve_cond_params(self):

        # ----- float/int parameter values -----
        mean_value = 2
        std_value = 0.5
        l = CondGaussianLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            cond_f=lambda data: {"mean": mean_value, "std": std_value},
        )

        for mean_node, std_node in zip(*l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())):
            self.assertTrue(mean_node == mean_value)
            self.assertTrue(std_node == std_value)

        # ----- list parameter values -----
        mean_values = [1.0, 5.0, -3.0]
        std_values = [0.25, 0.5, 0.3]
        l.set_cond_f(lambda data: {"mean": mean_values, "std": std_values})

        for mean_actual, std_actual, mean_node, std_node in zip(
            mean_values, std_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        ):
            self.assertTrue(mean_actual == mean_node)
            self.assertTrue(std_actual == std_node)

        # wrong number of values
        l.set_cond_f(lambda data: {"mean": mean_values[:-1], "std": std_values})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())
        l.set_cond_f(lambda data: {"mean": mean_values, "std": std_values[:-1]})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": [mean_values for _ in range(3)],
                "std": std_values,
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())
        l.set_cond_f(
            lambda data: {
                "mean": mean_values,
                "std": [std_values for _ in range(3)],
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # ----- numpy parameter values -----
        l.set_cond_f(
            lambda data: {
                "mean": tl.tensor(mean_values),
                "std": tl.tensor(std_values),
            }
        )
        for mean_actual, std_actual, mean_node, std_node in zip(
            mean_values, std_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
        ):
            self.assertTrue(mean_node == mean_actual)
            self.assertTrue(std_node == std_actual)

        # wrong number of values
        l.set_cond_f(
            lambda data: {
                "mean": tl.tensor(mean_values[:-1]),
                "std": tl.tensor(std_values),
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())
        l.set_cond_f(
            lambda data: {
                "mean": tl.tensor(mean_values),
                "std": tl.tensor(std_values[:-1]),
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": tl.tensor([mean_values for _ in range(3)]),
                "std": tl.tensor(std_values),
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())
        l.set_cond_f(
            lambda data: {
                "mean": tl.tensor(mean_values),
                "std": tl.tensor([std_values for _ in range(3)]),
            }
        )
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(CondGaussianLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGaussianLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(mean=-1.0, std=1.5)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian(mean=1.0, std=0.5)]),
            ]
        )
        self.assertTrue(isinstance(gaussian, CondGaussianLayer))
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGaussian))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGaussianLayer))
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
